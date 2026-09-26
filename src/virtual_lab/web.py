"""Requests to scientific databases, made on behalf of a model.

Every request here originates from arguments a model chose, so the destination is treated as
untrusted even though the URL is assembled from a fixed template. Tools never accept a URL: they
accept an identifier or a query, and this module builds the address, checks it against a list of
hosts the library is willing to talk to, and checks it again at every redirect.

The rest of the module exists because a meeting is long and an external service is not obliged to
cooperate. A request that hangs stalls a meeting, a service that rate limits will refuse the next
several calls, and a response with no size limit can exhaust memory or fill a context window with
one document.
"""

import json
import random
import threading
import time
import urllib.parse
from collections import OrderedDict
from typing import Any

import requests

from virtual_lab.constants import (
    MAX_RESPONSE_BYTES,
    MAX_WEB_CACHE_ENTRIES,
    MIN_SECONDS_BETWEEN_REQUESTS,
    WEB_BACKOFF_SECONDS,
    WEB_MAX_ATTEMPTS,
    WEB_MAX_REDIRECTS,
    WEB_TIMEOUT_SECONDS,
    WEB_USER_AGENT,
)

# The hosts this library will talk to. A model chooses the arguments to every request made here,
# so the set of places those requests can reach is fixed in code rather than derived from them.
# Adding an entry is a deliberate decision to trust a service with queries drawn from a meeting.
ALLOWED_HOSTS = frozenset(
    {
        # NCBI: PubMed, PubMed Central, PubChem
        "eutils.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "pubchem.ncbi.nlm.nih.gov",
        # EMBL-EBI: UniProt, AlphaFold, Europe PMC, ChEMBL
        "rest.uniprot.org",
        "alphafold.ebi.ac.uk",
        "www.ebi.ac.uk",
        # RCSB: Protein Data Bank
        "data.rcsb.org",
        "search.rcsb.org",
        "files.rcsb.org",
        # Preprints
        "export.arxiv.org",
        "api.biorxiv.org",
    }
)

# Statuses worth trying again. A rate limit and a gateway error are both temporary; a 404 means
# the identifier was wrong and asking again will not change that.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class WebRequestError(Exception):
    """Raised when a request to an external service cannot be completed."""


class DisallowedHostError(WebRequestError):
    """Raised when a request would go somewhere this library does not talk to."""


class ResponseTooLargeError(WebRequestError):
    """Raised when a service returns more data than a meeting can be given."""


def check_host(url: str) -> str:
    """Checks that a URL points somewhere this library is willing to talk to.

    :param url: The URL to check.
    :raises DisallowedHostError: If the scheme is not HTTPS or the host is not allowed.
    :return: The URL, unchanged.
    """
    parsed = urllib.parse.urlsplit(url)

    # Plain HTTP would let anything between here and the service read and rewrite the exchange,
    # and every service in the list below speaks HTTPS
    if parsed.scheme != "https":
        raise DisallowedHostError(f'Refusing to request "{url}": only https is allowed')

    if parsed.hostname is None:
        raise DisallowedHostError(f'Refusing to request "{url}": no host')

    if parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise DisallowedHostError(
            f'Refusing to request "{url}": {parsed.hostname} is not an allowed host. '
            f"Allowed hosts: {', '.join(sorted(ALLOWED_HOSTS))}."
        )

    return url


def encode_segment(value: str) -> str:
    """Encodes a value for use as a single path segment.

    Slashes are encoded rather than passed through, so an identifier a model invented cannot
    add path segments of its own or climb out of the path it was given.

    :param value: The value to encode.
    :raises ValueError: If the value is empty.
    :return: The encoded value.
    """
    if not value or not value.strip():
        raise ValueError("A path segment may not be empty")

    return urllib.parse.quote(value.strip(), safe="")


def build_url(template: str, **segments: str) -> str:
    """Builds a URL from a fixed template and encoded path segments.

    The template is written in code and the segments come from a meeting, which is why only the
    segments are substituted and each is encoded first. The result is checked before it is
    returned, so a template mistake cannot produce a request to somewhere unintended.

    :param template: A format string containing the scheme, host, and path.
    :param segments: The values to substitute, each encoded as a single path segment.
    :raises DisallowedHostError: If the resulting URL is not allowed.
    :raises ValueError: If a segment is empty.
    :return: The URL.
    """
    return check_host(
        template.format(**{name: encode_segment(value) for name, value in segments.items()})
    )


class RateLimiter:
    """Keeps requests to each host to a polite rate.

    Public scientific APIs ask for this and enforce it: NCBI limits unauthenticated clients to a
    few requests a second and will refuse the rest. A meeting making several tool calls in a row
    would otherwise spend its retries on self-inflicted rate limits.
    """

    def __init__(self, min_interval: float = MIN_SECONDS_BETWEEN_REQUESTS) -> None:
        self.min_interval = min_interval
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        """Sleeps if the last request to this host was too recent.

        :param host: The host about to be contacted.
        """
        with self._lock:
            elapsed = time.monotonic() - self._last_request.get(host, float("-inf"))
            delay = self.min_interval - elapsed

            if delay > 0:
                time.sleep(delay)

            self._last_request[host] = time.monotonic()


class ResponseCache:
    """Remembers responses for the life of the process.

    Agents ask the same question more than once, both within a meeting and across the meetings of
    a project, and every repeat is a request some service serves for free. This is deliberately
    only held in memory: a cache on disk would need an expiry policy, and quietly returning last
    week's answer to a question about a database is worse than asking again.
    """

    def __init__(self, max_entries: int = MAX_WEB_CACHE_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        """Returns a cached response, or None if it is not held."""
        with self._lock:
            if key not in self._entries:
                return None

            self._entries.move_to_end(key)

            return self._entries[key]

    def put(self, key: str, value: str) -> None:
        """Stores a response, discarding the least recently used if the cache is full."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Forgets everything, so that a caller can guarantee a fresh request."""
        with self._lock:
            self._entries.clear()


RATE_LIMITER = RateLimiter()
RESPONSE_CACHE = ResponseCache()


def cache_key(url: str, params: dict[str, Any] | None) -> str:
    """Builds the key a response is cached under.

    :param url: The URL requested.
    :param params: The query parameters, if any.
    :return: A key that distinguishes requests differing only in their parameters.
    """
    return f"{url}?{urllib.parse.urlencode(sorted((params or {}).items()))}"


def read_capped(response: requests.Response, max_bytes: int) -> str:
    """Reads a response body, refusing to read more than a meeting can be given.

    The body is read in chunks and abandoned once it is too long, rather than read in full and
    then measured, since the point is not to hold the whole of an unbounded response in memory.

    :param response: The streaming response to read.
    :param max_bytes: The most bytes to accept.
    :raises ResponseTooLargeError: If the body exceeds the limit.
    :return: The decoded body.
    """
    declared = response.headers.get("Content-Length")

    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise ResponseTooLargeError(
            f"{response.url} declared {int(declared):,} bytes, over the {max_bytes:,} byte limit"
        )

    chunks: list[bytes] = []
    total = 0

    for chunk in response.iter_content(chunk_size=8192):
        total += len(chunk)

        if total > max_bytes:
            raise ResponseTooLargeError(
                f"{response.url} returned more than the {max_bytes:,} byte limit"
            )

        chunks.append(chunk)

    return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    """Decides how long to wait before trying a request again.

    A service that says how long to wait is believed, because it knows and because ignoring it
    is how a client earns a longer refusal.

    :param attempt: Which attempt has just failed, counting from one.
    :param response: The response that failed, if there was one.
    :return: Seconds to wait.
    """
    if response is not None:
        requested = response.headers.get("Retry-After")

        if requested is not None and requested.strip().isdigit():
            return min(float(requested.strip()), 60.0)

    # Jitter keeps several tool calls that failed together from retrying in lockstep
    return WEB_BACKOFF_SECONDS * (2 ** (attempt - 1)) * (1 + random.random() * 0.1)


class RetryableResponse(Exception):
    """Raised internally when a response should be requested again."""

    def __init__(self, detail: str, response: requests.Response | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.response = response


def request_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = WEB_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_attempts: int = WEB_MAX_ATTEMPTS,
    use_cache: bool = True,
) -> str:
    """Fetches a URL, returning the body as text.

    :param url: The URL to fetch, which must be on an allowed host.
    :param params: Query parameters.
    :param headers: Additional request headers.
    :param timeout: Seconds to allow for the connection and for each read.
    :param max_bytes: The most bytes to accept in the response.
    :param max_attempts: Attempts to make before giving up, counting the first.
    :param use_cache: Whether an identical earlier response may be reused.
    :raises DisallowedHostError: If the URL, or a redirect from it, is not allowed.
    :raises ResponseTooLargeError: If the response exceeds max_bytes.
    :raises WebRequestError: If the request fails after all attempts.
    :return: The response body.
    """
    check_host(url)

    key = cache_key(url, params)

    if use_cache and (cached := RESPONSE_CACHE.get(key)) is not None:
        return cached

    request_headers = {"User-Agent": WEB_USER_AGENT, **(headers or {})}
    body: str | None = None
    last_error: str = "no attempts were made"

    for attempt in range(1, max_attempts + 1):
        try:
            body = follow(
                url=url,
                params=params,
                headers=request_headers,
                timeout=timeout,
                max_bytes=max_bytes,
            )
            break
        except RetryableResponse as retryable:
            last_error = retryable.detail

            if attempt == max_attempts:
                break

            time.sleep(retry_delay(attempt=attempt, response=retryable.response))
        except requests.RequestException as error:
            last_error = f"{type(error).__name__}: {error}"

            if attempt == max_attempts:
                break

            time.sleep(retry_delay(attempt=attempt, response=None))

    if body is None:
        raise WebRequestError(
            f"Could not fetch {url} after {max_attempts} "
            f"attempt{'s' if max_attempts > 1 else ''}: {last_error}"
        )

    if use_cache:
        RESPONSE_CACHE.put(key, body)

    return body


def follow(
    url: str,
    params: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
) -> str:
    """Makes one request, following redirects only to hosts that are allowed.

    Redirects are followed by hand because the alternative is not safe here. A client that
    follows them automatically checks the host it was asked about and not the host it ends up
    talking to, which turns one allowed address into a request to anywhere.

    :param url: The URL to fetch.
    :param params: Query parameters, sent to the first URL only.
    :param headers: Request headers.
    :param timeout: Seconds to allow for the connection and for each read.
    :param max_bytes: The most bytes to accept.
    :raises DisallowedHostError: If a redirect leads somewhere not allowed.
    :raises RetryableResponse: If the service asked to be tried again.
    :raises WebRequestError: If the service refused the request.
    :return: The response body.
    """
    current = url
    query = params

    for _ in range(WEB_MAX_REDIRECTS + 1):
        check_host(current)
        RATE_LIMITER.wait(urllib.parse.urlsplit(current).hostname or "")

        with requests.get(
            current,
            params=query,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        ) as response:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")

                if not location:
                    raise WebRequestError(f"{current} returned {response.status_code} with no location")

                # A relative location resolves against the current URL, which keeps it on the
                # same host; an absolute one is checked on the next pass round this loop
                current = urllib.parse.urljoin(current, location)
                query = None
                continue

            if response.status_code in RETRYABLE_STATUSES:
                raise RetryableResponse(
                    detail=f"{current} returned {response.status_code}", response=response
                )

            if not response.ok:
                raise WebRequestError(f"{current} returned {response.status_code}")

            return read_capped(response=response, max_bytes=max_bytes)

    raise WebRequestError(f"{url} redirected more than {WEB_MAX_REDIRECTS} times")


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = WEB_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_attempts: int = WEB_MAX_ATTEMPTS,
    use_cache: bool = True,
) -> Any:
    """Fetches a URL and parses the body as JSON.

    :param url: The URL to fetch, which must be on an allowed host.
    :param params: Query parameters.
    :param headers: Additional request headers.
    :param timeout: Seconds to allow for the connection and for each read.
    :param max_bytes: The most bytes to accept in the response.
    :param max_attempts: Attempts to make before giving up, counting the first.
    :param use_cache: Whether an identical earlier response may be reused.
    :raises WebRequestError: If the request fails, or the body is not JSON.
    :return: The parsed body.
    """
    body = request_text(
        url=url,
        params=params,
        headers={"Accept": "application/json", **(headers or {})},
        timeout=timeout,
        max_bytes=max_bytes,
        max_attempts=max_attempts,
        use_cache=use_cache,
    )

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise WebRequestError(f"{url} did not return JSON: {error}") from error


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = WEB_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_attempts: int = WEB_MAX_ATTEMPTS,
) -> Any:
    """Sends a JSON body to a URL and parses the response as JSON.

    Some search APIs take a structured query that will not fit in a query string. Redirects are
    not followed for these, since replaying a body to a new address is not something to do
    quietly, and no allowed service needs it.

    :param url: The URL to post to, which must be on an allowed host.
    :param payload: The JSON body to send.
    :param headers: Additional request headers.
    :param timeout: Seconds to allow for the connection and for each read.
    :param max_bytes: The most bytes to accept in the response.
    :param max_attempts: Attempts to make before giving up, counting the first.
    :raises WebRequestError: If the request fails, or the body is not JSON.
    :return: The parsed body.
    """
    check_host(url)

    request_headers = {
        "User-Agent": WEB_USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        **(headers or {}),
    }
    last_error = "no attempts were made"

    for attempt in range(1, max_attempts + 1):
        RATE_LIMITER.wait(urllib.parse.urlsplit(url).hostname or "")

        try:
            with requests.post(
                url,
                json=payload,
                headers=request_headers,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.status_code in RETRYABLE_STATUSES:
                    last_error = f"{url} returned {response.status_code}"

                    if attempt == max_attempts:
                        break

                    time.sleep(retry_delay(attempt=attempt, response=response))
                    continue

                if not response.ok:
                    raise WebRequestError(f"{url} returned {response.status_code}")

                body = read_capped(response=response, max_bytes=max_bytes)

            try:
                return json.loads(body)
            except json.JSONDecodeError as error:
                raise WebRequestError(f"{url} did not return JSON: {error}") from error
        except requests.RequestException as error:
            last_error = f"{type(error).__name__}: {error}"

            if attempt == max_attempts:
                break

            time.sleep(retry_delay(attempt=attempt, response=None))

    raise WebRequestError(
        f"Could not post to {url} after {max_attempts} "
        f"attempt{'s' if max_attempts > 1 else ''}: {last_error}"
    )
