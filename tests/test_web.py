"""Tests for requests made on behalf of a model.

These run offline. The transport is faked so that redirects, rate limits, oversized bodies, and
retries can be tested deterministically, none of which a real service can be asked to produce on
demand. A handful of tests do hit the real services and are skipped unless VIRTUAL_LAB_LIVE_TESTS
is set, so the default suite stays fast and does not depend on a network or on someone else's
uptime.
"""

import os
import threading
import time
import urllib.parse

import pytest
import requests
from requests.structures import CaseInsensitiveDict

import virtual_lab.web as web
from virtual_lab.constants import MIN_SECONDS_BETWEEN_REQUESTS
from virtual_lab.web import (
    ALLOWED_HOSTS,
    RESPONSE_CACHE,
    DisallowedHostError,
    RateLimiter,
    ResponseCache,
    ResponseTooLargeError,
    WebRequestError,
    build_url,
    cache_key,
    check_host,
    encode_segment,
    request_json,
    request_text,
    retry_delay,
)

# Captured before any test can stub it out, for the one test that needs to really wait
REAL_SLEEP = time.sleep

live_only = pytest.mark.skipif(
    os.environ.get("VIRTUAL_LAB_LIVE_TESTS") != "1",
    reason="Set VIRTUAL_LAB_LIVE_TESTS=1 to query the real services",
)

UNIPROT = "https://rest.uniprot.org/uniprotkb/P01308.json"
RCSB = "https://search.rcsb.org/rcsbsearch/v2/query"


class FakeResponse:
    """Stands in for a streaming requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        body: bytes = b"{}",
        headers: dict[str, str] | None = None,
        url: str = UNIPROT,
    ) -> None:
        self.status_code = status_code
        self.body = body
        # Case-insensitive, as real requests is. A plain dict would let production code read a
        # header under the wrong case and still pass here.
        self.headers = CaseInsensitiveDict(headers or {})
        self.url = url

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def is_redirect(self) -> bool:
        return self.status_code in {301, 302, 303, 307, 308} and "Location" in self.headers

    @property
    def is_permanent_redirect(self) -> bool:
        return self.status_code in {301, 308} and "Location" in self.headers

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture(autouse=True)
def fast_and_isolated(monkeypatch: pytest.MonkeyPatch):
    """Clears the cache between tests and removes the deliberate delays."""
    RESPONSE_CACHE.clear()
    monkeypatch.setattr(web.RATE_LIMITER, "min_interval", 0.0)
    monkeypatch.setattr(web.time, "sleep", lambda seconds: None)

    yield

    RESPONSE_CACHE.clear()


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    """Replaces the HTTP layer, recording requests and replaying queued responses."""

    class Transport:
        def __init__(self) -> None:
            self.responses: list[FakeResponse] = []
            self.requests: list[dict] = []

        def get(self, url, **kwargs):
            self.requests.append({"url": url, **kwargs})

            if not self.responses:
                return FakeResponse(url=url)

            return self.responses.pop(0)

        @property
        def urls(self) -> list[str]:
            return [request["url"] for request in self.requests]

    fake = Transport()
    monkeypatch.setattr(web.requests, "get", fake.get)

    return fake


class TestAllowedDestinations:
    """A model chooses the arguments to every request, so the destination is fixed in code."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://rest.uniprot.org/x",
            "https://evil.example.com/x",
            "https://rest.uniprot.org.evil.com/x",
            "https://evil.com/?host=rest.uniprot.org",
            "https://user:password@evil.com/x",
            "https://127.0.0.1/x",
            "https://localhost/x",
            "https://169.254.169.254/latest/meta-data/",
            "https://[::1]/x",
            "https://10.0.0.1/x",
            "file:///etc/passwd",
            "ftp://rest.uniprot.org/x",
            "https:///no-host",
        ],
    )
    def test_a_destination_outside_the_list_is_refused(self, url: str) -> None:
        with pytest.raises(DisallowedHostError):
            check_host(url)

    @pytest.mark.parametrize(
        "url",
        [
            # Ends in an allowed host without being one. This is what an allowlist implemented
            # with endswith would let through, and the reason membership is tested exactly.
            "https://evilrest.uniprot.org/x",
            "https://notdata.rcsb.org/x",
            # A subdomain of an allowed host is not the service that was vouched for
            "https://x.rest.uniprot.org/x",
            "https://files.rcsb.org.attacker.net/x",
            # A trailing dot is a distinct name to a resolver
            "https://rest.uniprot.org./x",
        ],
    )
    def test_a_near_miss_of_an_allowed_host_is_refused(self, url: str) -> None:
        with pytest.raises(DisallowedHostError):
            check_host(url)

    @pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS))
    def test_every_listed_host_is_accepted(self, host: str) -> None:
        assert check_host(f"https://{host}/path") == f"https://{host}/path"

    def test_a_listed_host_is_matched_case_insensitively(self) -> None:
        assert check_host("https://REST.UniProt.ORG/x")

    def test_the_error_names_the_hosts_that_are_allowed(self) -> None:
        with pytest.raises(DisallowedHostError, match="rest.uniprot.org"):
            check_host("https://evil.com/x")

    def test_the_list_is_exactly_this_set(self) -> None:
        # Pinned as a set rather than by a property of the names. Asserting something like the
        # top level domain instead would admit any general purpose host under .org, which is the
        # one thing this list must never hold, while failing the build for a legitimate database
        # under some other domain. Adding a service should be a visible change to this test.
        assert ALLOWED_HOSTS == frozenset(
            {
                "eutils.ncbi.nlm.nih.gov",
                "www.ncbi.nlm.nih.gov",
                "pubchem.ncbi.nlm.nih.gov",
                "rest.uniprot.org",
                "alphafold.ebi.ac.uk",
                "www.ebi.ac.uk",
                "data.rcsb.org",
                "search.rcsb.org",
                "files.rcsb.org",
                "export.arxiv.org",
                "api.biorxiv.org",
            }
        )


class TestWhatIsCheckedIsWhatIsSent:
    """The URL is validated by one library and requested by another, which must agree.

    They do not agree by default. CPython's urlsplit ends the authority at a slash, question
    mark, or hash, while urllib3 also ends it at a backslash, so a URL exists that one reads as
    an allowed host and the other dials as an entirely different address. check_host therefore
    normalizes through the requesting library and returns the URL to request.
    """

    @pytest.mark.parametrize(
        "url, dialed",
        [
            ("https://169.254.169.254\\@rest.uniprot.org/latest/meta-data/", "169.254.169.254"),
            ("https://127.0.0.1\\@rest.uniprot.org/x", "127.0.0.1"),
            ("https://evil.com\\@data.rcsb.org/x", "evil.com"),
        ],
    )
    def test_a_backslash_authority_cannot_smuggle_a_host(self, url: str, dialed: str) -> None:
        # Guard the premise first: if requests ever stops reading these as the other host, this
        # test would pass for a reason that has nothing to do with check_host
        prepared = requests.models.PreparedRequest()
        prepared.prepare_url(url, None)
        assert urllib.parse.urlsplit(prepared.url).hostname == dialed
        assert urllib.parse.urlsplit(url).hostname == "rest.uniprot.org" or dialed == "evil.com"

        with pytest.raises(DisallowedHostError):
            check_host(url)

    def test_the_url_returned_is_the_url_to_request(self) -> None:
        # Callers must use the return value, not the argument, or the check applies to a
        # different address than the request does
        assert check_host("https://REST.UNIPROT.ORG/x") == "https://rest.uniprot.org/x"

    def test_normalizing_is_stable(self) -> None:
        # Requesting the checked URL re-prepares it, so a second pass must not change it again
        once = check_host("https://rest.uniprot.org/uniprotkb/P01308.json")
        assert web.normalize_url(once) == once

    @pytest.mark.parametrize(
        "url",
        [
            "https://user:password@rest.uniprot.org/x",
            "https://token@data.rcsb.org/x",
        ],
    )
    def test_credentials_are_refused_even_on_an_allowed_host(self, url: str) -> None:
        with pytest.raises(DisallowedHostError, match="credentials"):
            check_host(url)

    @pytest.mark.parametrize("url", ["https://[::1x]/x", "https://[not-an-address]/x"])
    def test_an_unparseable_host_is_refused_as_a_host_error(self, url: str) -> None:
        # Not a bare ValueError: no caller of this module is prepared for one, so it would
        # escape every handler and end a meeting instead of being reported as a bad destination
        with pytest.raises(DisallowedHostError):
            check_host(url)


class TestIdentifiersCannotReshapeAUrl:
    @pytest.mark.parametrize(
        "identifier",
        [
            "../../../etc/passwd",
            "P01308/../../admin",
            "P01308?format=txt",
            "P01308#fragment",
            "P01308&x=1",
            "with space",
            "semi;colon",
            "at@sign",
        ],
    )
    def test_a_separator_is_encoded_rather_than_honoured(self, identifier: str) -> None:
        url = build_url(
            "https://rest.uniprot.org/uniprotkb/{accession}.json", accession=identifier
        )

        assert url.startswith("https://rest.uniprot.org/uniprotkb/")
        for character in "/?#&":
            assert character not in url[len("https://rest.uniprot.org/uniprotkb/") :]

    def test_an_empty_identifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="may not be empty"):
            encode_segment("   ")

    @pytest.mark.parametrize("identifier", [".", "..", "  ..  "])
    def test_a_relative_path_segment_is_refused(self, identifier: str) -> None:
        # A dot is unreserved, so encoding leaves these intact and the server resolves them
        # against the path the template chose. Encoding alone does not make them harmless.
        with pytest.raises(ValueError, match="relative path segment"):
            encode_segment(identifier)

    def test_a_dotted_identifier_that_is_not_relative_is_kept(self) -> None:
        assert encode_segment("P01308.json") == "P01308.json"
        assert encode_segment("...") == "..."

    def test_a_substituted_host_is_still_checked(self) -> None:
        # Defence in depth: a mistake in a template cannot open a hole
        with pytest.raises(DisallowedHostError):
            build_url("https://{host}/x", host="evil.com")

    def test_surrounding_whitespace_is_dropped(self) -> None:
        assert encode_segment("  P01308  ") == "P01308"


class TestRedirects:
    def test_a_redirect_within_the_list_is_followed(self, transport) -> None:
        transport.responses = [
            FakeResponse(status_code=301, headers={"Location": "https://rest.uniprot.org/moved"}),
            FakeResponse(body=b'{"ok": true}'),
        ]

        assert request_json(UNIPROT) == {"ok": True}
        assert transport.urls[1] == "https://rest.uniprot.org/moved"

    def test_a_redirect_off_the_list_is_refused(self, transport) -> None:
        # A client that follows redirects itself checks the host it was asked about, not the host
        # it ends up talking to, which turns one allowed address into a request to anywhere
        transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "https://evil.com/steal"})
        ]

        with pytest.raises(DisallowedHostError, match="evil.com"):
            request_text(UNIPROT)

        assert "https://evil.com/steal" not in transport.urls

    def test_a_relative_redirect_stays_on_the_same_host(self, transport) -> None:
        transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "/elsewhere"}),
            FakeResponse(body=b'{"ok": true}'),
        ]

        request_json(UNIPROT)

        assert transport.urls[1] == "https://rest.uniprot.org/elsewhere"

    def test_a_protocol_downgrade_by_redirect_is_refused(self, transport) -> None:
        transport.responses = [
            FakeResponse(status_code=301, headers={"Location": "http://rest.uniprot.org/x"})
        ]

        with pytest.raises(DisallowedHostError, match="only https"):
            request_text(UNIPROT)

    def test_a_redirect_loop_gives_up(self, transport) -> None:
        transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "https://rest.uniprot.org/loop"})
            for _ in range(20)
        ]

        with pytest.raises(WebRequestError, match="redirected more than"):
            request_text(UNIPROT)

    def test_query_parameters_are_not_replayed_to_the_new_address(self, transport) -> None:
        transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "https://rest.uniprot.org/moved"}),
            FakeResponse(body=b"{}"),
        ]

        request_json(UNIPROT, params={"size": 5})

        assert transport.requests[0]["params"] == {"size": 5}
        assert transport.requests[1]["params"] is None

    def test_the_transport_is_told_not_to_follow_redirects_itself(self, transport) -> None:
        request_text(UNIPROT)

        assert transport.requests[0]["allow_redirects"] is False

    def test_a_redirect_with_no_location_is_not_read_as_a_body(self, transport) -> None:
        # is_redirect is false without a Location header, and ok() is true below 400, so a 3xx
        # tested that way falls through and its body is returned and cached as the resource
        transport.responses = [FakeResponse(status_code=302, body=b"<h1>moved</h1>")]

        with pytest.raises(WebRequestError, match="no location"):
            request_text(UNIPROT)

    def test_a_not_modified_response_is_not_read_as_a_body(self, transport) -> None:
        transport.responses = [FakeResponse(status_code=304, body=b"")]

        with pytest.raises(WebRequestError):
            request_text(UNIPROT, headers={"If-None-Match": "abc"})

    def test_a_backslash_in_a_location_cannot_escape_the_allowlist(self, transport) -> None:
        transport.responses = [
            FakeResponse(
                status_code=302,
                headers={"Location": "https://169.254.169.254\\@rest.uniprot.org/x"},
            )
        ]

        with pytest.raises(DisallowedHostError):
            request_text(UNIPROT)

        assert len(transport.requests) == 1

    def test_a_scheme_relative_location_is_checked(self, transport) -> None:
        transport.responses = [FakeResponse(status_code=302, headers={"Location": "//evil.com/x"})]

        with pytest.raises(DisallowedHostError, match="evil.com"):
            request_text(UNIPROT)


class TestResponseSize:
    def test_a_body_over_the_limit_is_refused(self, transport) -> None:
        transport.responses = [FakeResponse(body=b"x" * 5000)]

        with pytest.raises(ResponseTooLargeError):
            request_text(UNIPROT, max_bytes=1000)

    def test_a_declared_length_over_the_limit_is_refused_before_reading(self, transport) -> None:
        transport.responses = [
            FakeResponse(body=b"x" * 5000, headers={"Content-Length": "5000"})
        ]

        with pytest.raises(ResponseTooLargeError, match="declared"):
            request_text(UNIPROT, max_bytes=1000)

    def test_a_body_at_the_limit_is_accepted(self, transport) -> None:
        transport.responses = [FakeResponse(body=b"x" * 1000)]

        assert len(request_text(UNIPROT, max_bytes=1000)) == 1000

    def test_an_undeclared_oversized_body_is_still_caught(self, transport) -> None:
        # A service that lies about or omits its length must not be able to overrun the limit
        transport.responses = [FakeResponse(body=b"x" * 100_000, headers={})]

        with pytest.raises(ResponseTooLargeError, match="more than"):
            request_text(UNIPROT, max_bytes=5000)

    def test_a_lowercase_length_header_is_still_read(self, transport) -> None:
        transport.responses = [FakeResponse(body=b"x" * 5000, headers={"content-length": "5000"})]

        with pytest.raises(ResponseTooLargeError, match="declared"):
            request_text(UNIPROT, max_bytes=1000)

    def test_an_unparseable_length_header_does_not_crash(self, transport) -> None:
        # A superscript passes isdigit but not int(), and headers arrive decoded as latin-1, so
        # such a character can reach here from the wire. It must not escape as a ValueError.
        transport.responses = [FakeResponse(body=b"hello", headers={"Content-Length": "\u00b2"})]

        assert request_text(UNIPROT) == "hello"

    def test_a_cached_body_is_still_measured_against_a_smaller_limit(self, transport) -> None:
        # The cache is consulted before the limit is applied, so without the limit in the key a
        # body stored under a generous one would be handed to a caller that asked for less
        transport.responses = [FakeResponse(body=b"x" * 5000), FakeResponse(body=b"x" * 5000)]

        assert len(request_text(UNIPROT, max_bytes=100_000)) == 5000

        with pytest.raises(ResponseTooLargeError):
            request_text(UNIPROT, max_bytes=1000)


class TestDecoding:
    def test_a_text_response_with_no_charset_is_read_as_utf_8(self, transport) -> None:
        # The client's own guess is ISO-8859-1 here, which is what the older HTTP specification
        # called for and what almost no service means. Reading a record that way is mojibake.
        transport.responses = [
            FakeResponse(
                body="alpha \u03b1 \u00c5ngstr\u00f6m".encode(),
                headers={"Content-Type": "text/plain"},
            )
        ]

        assert request_text(UNIPROT) == "alpha \u03b1 \u00c5ngstr\u00f6m"

    def test_a_declared_charset_is_honoured(self, transport) -> None:
        transport.responses = [
            FakeResponse(
                body="caf\u00e9".encode("iso-8859-1"),
                headers={"Content-Type": "text/plain; charset=iso-8859-1"},
            )
        ]

        assert request_text(UNIPROT) == "caf\u00e9"

    def test_a_charset_python_does_not_have_does_not_crash(self, transport) -> None:
        # LookupError is neither a WebRequestError nor a RequestException, so it would escape
        # every handler in this module and in its callers
        transport.responses = [
            FakeResponse(
                body=b'{"a": 1}',
                headers={"Content-Type": "application/json; charset=unknown-8bit"},
            )
        ]

        assert request_json(UNIPROT) == {"a": 1}

    def test_a_quoted_charset_is_read(self, transport) -> None:
        transport.responses = [
            FakeResponse(body=b"ok", headers={"Content-Type": 'text/plain; charset="utf-8"'})
        ]

        assert request_text(UNIPROT) == "ok"


class TestRetries:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_a_temporary_failure_is_retried(self, transport, status: int) -> None:
        transport.responses = [
            FakeResponse(status_code=status),
            FakeResponse(body=b'{"ok": true}'),
        ]

        assert request_json(UNIPROT) == {"ok": True}
        assert len(transport.requests) == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_permanent_failure_is_not_retried(self, transport, status: int) -> None:
        # A wrong identifier will still be wrong the second time, and asking again is rude
        transport.responses = [FakeResponse(status_code=status) for _ in range(5)]

        with pytest.raises(WebRequestError, match=str(status)):
            request_text(UNIPROT)

        assert len(transport.requests) == 1

    def test_a_connection_error_is_retried(self, transport, monkeypatch) -> None:
        attempts = {"count": 0}

        def flaky(url, **kwargs):
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise requests.ConnectionError("network down")

            return FakeResponse(body=b'{"ok": true}')

        monkeypatch.setattr(web.requests, "get", flaky)

        assert request_json(UNIPROT) == {"ok": True}
        assert attempts["count"] == 3

    def test_attempts_are_bounded(self, transport) -> None:
        transport.responses = [FakeResponse(status_code=503) for _ in range(10)]

        with pytest.raises(WebRequestError, match="after 3 attempts"):
            request_text(UNIPROT)

        assert len(transport.requests) == 3

    def test_the_error_says_what_went_wrong(self, transport) -> None:
        transport.responses = [FakeResponse(status_code=503) for _ in range(3)]

        with pytest.raises(WebRequestError, match="503"):
            request_text(UNIPROT)

    def test_a_service_asking_for_a_delay_is_believed(self) -> None:
        response = FakeResponse(status_code=429, headers={"Retry-After": "7"})

        assert retry_delay(attempt=1, response=response) == 7.0

    def test_an_absurd_delay_request_is_capped(self) -> None:
        # A service asking for an hour would hold a meeting open for an hour
        response = FakeResponse(status_code=429, headers={"Retry-After": "86400"})

        assert retry_delay(attempt=1, response=response) == 60.0

    def test_backoff_grows_between_attempts(self) -> None:
        first = retry_delay(attempt=1, response=None)
        third = retry_delay(attempt=3, response=None)

        assert third > first

    def test_a_nonsense_delay_request_is_ignored(self) -> None:
        response = FakeResponse(status_code=429, headers={"Retry-After": "Tue, 1 Jan 2030"})

        assert retry_delay(attempt=1, response=response) < 10


class TestCaching:
    def test_a_repeated_request_is_served_from_memory(self, transport) -> None:
        transport.responses = [FakeResponse(body=b'{"n": 1}')]

        assert request_json(UNIPROT) == {"n": 1}
        assert request_json(UNIPROT) == {"n": 1}
        assert len(transport.requests) == 1

    def test_different_parameters_are_different_requests(self, transport) -> None:
        transport.responses = [FakeResponse(body=b'{"n": 1}'), FakeResponse(body=b'{"n": 2}')]

        request_json(UNIPROT, params={"size": 1})
        request_json(UNIPROT, params={"size": 2})

        assert len(transport.requests) == 2

    def test_parameter_order_does_not_change_the_key(self) -> None:
        assert cache_key(UNIPROT, {"a": 1, "b": 2}) == cache_key(UNIPROT, {"b": 2, "a": 1})

    def test_a_different_accept_header_is_a_different_request(self) -> None:
        # Several allowed services vary their answer by Accept, so a body fetched as text must
        # not be served to a caller that asked for JSON
        assert cache_key(UNIPROT, None, headers={"Accept": "text/plain"}) != cache_key(
            UNIPROT, None, headers={"Accept": "application/json"}
        )

    def test_a_different_size_limit_is_a_different_request(self) -> None:
        assert cache_key(UNIPROT, None, max_bytes=100) != cache_key(UNIPROT, None, max_bytes=200)

    def test_text_and_json_do_not_share_an_entry(self, transport) -> None:
        transport.responses = [
            FakeResponse(body=b"ID   INS_HUMAN", headers={"Content-Type": "text/plain"}),
            FakeResponse(body=b'{"ok": true}'),
        ]

        assert request_text(UNIPROT) == "ID   INS_HUMAN"
        assert request_json(UNIPROT) == {"ok": True}
        assert len(transport.requests) == 2

    def test_caching_can_be_declined_for_reading_as_well_as_writing(self, transport) -> None:
        # Two requests that both decline the cache would pass even if only the write were
        # guarded, since nothing was ever stored. The case that matters is a warm cache.
        transport.responses = [FakeResponse(body=b'{"n": 1}'), FakeResponse(body=b'{"n": 2}')]

        assert request_json(UNIPROT) == {"n": 1}
        assert request_json(UNIPROT, use_cache=False) == {"n": 2}
        assert len(transport.requests) == 2

    def test_a_failure_is_not_cached(self, transport) -> None:
        transport.responses = [
            *[FakeResponse(status_code=503) for _ in range(3)],
            FakeResponse(body=b'{"ok": true}'),
        ]

        with pytest.raises(WebRequestError):
            request_text(UNIPROT)

        assert request_json(UNIPROT) == {"ok": True}

    def test_the_cache_discards_the_least_recently_used(self) -> None:
        cache = ResponseCache(max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")
        cache.put("c", "3")

        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("c") == "3"

    def test_the_cache_is_bounded_by_size_as_well_as_by_count(self) -> None:
        # An entry limit alone is not a bound on memory: it permits every entry to be the
        # largest response the size cap allows, which is over a gigabyte at the shipped values
        cache = ResponseCache(max_entries=100, max_characters=10)
        cache.put("a", "x" * 6)
        cache.put("b", "y" * 6)

        assert cache.get("a") is None
        assert cache.get("b") == "y" * 6

    def test_replacing_an_entry_does_not_double_count_it(self) -> None:
        cache = ResponseCache(max_entries=10, max_characters=10)
        cache.put("a", "x" * 6)
        cache.put("a", "y" * 6)
        cache.put("b", "z" * 4)

        assert cache.get("a") == "y" * 6
        assert cache.get("b") == "z" * 4


class TestRateLimiting:
    def test_requests_to_one_host_are_spaced_out(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(web.time, "sleep", slept.append)

        limiter = RateLimiter(min_interval=0.5)
        limiter.wait("rest.uniprot.org")
        limiter.wait("rest.uniprot.org")

        assert len(slept) == 1
        assert 0 < slept[0] <= 0.5

    def test_different_hosts_do_not_wait_on_each_other(self, monkeypatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(web.time, "sleep", slept.append)

        limiter = RateLimiter(min_interval=0.5)
        limiter.wait("rest.uniprot.org")
        limiter.wait("data.rcsb.org")

        assert slept == []

    def test_the_request_path_actually_waits(self, transport, monkeypatch) -> None:
        # The class above can be correct while nothing calls it. Deleting the call from the
        # request path is invisible to every other test here, and the consequence is external:
        # NCBI refuses an unauthenticated client that exceeds its limit.
        waited: list[str] = []
        monkeypatch.setattr(web.RATE_LIMITER, "wait", waited.append)

        request_json(UNIPROT)

        assert waited == ["rest.uniprot.org"]

    def test_every_redirect_hop_waits(self, transport, monkeypatch) -> None:
        waited: list[str] = []
        monkeypatch.setattr(web.RATE_LIMITER, "wait", waited.append)
        transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "https://data.rcsb.org/x"}),
            FakeResponse(body=b"{}"),
        ]

        request_json(UNIPROT)

        assert waited == ["rest.uniprot.org", "data.rcsb.org"]

    def test_a_sleeping_host_does_not_hold_up_another(self, monkeypatch) -> None:
        # Sleeping while holding the lock would make every host queue behind whichever one
        # happened to be waiting, which is the opposite of a per-host limit
        monkeypatch.setattr(web.time, "sleep", REAL_SLEEP)

        limiter = RateLimiter(min_interval=0.4)
        limiter.wait("busy.example")

        # Started first and given a moment to reach its sleep, so that the host below is
        # measured while the other is definitely waiting rather than racing it for the lock
        waiter = threading.Thread(target=limiter.wait, args=("busy.example",))
        waiter.start()
        REAL_SLEEP(0.05)

        started = time.monotonic()
        limiter.wait("idle.example")
        elapsed = time.monotonic() - started

        waiter.join()

        assert elapsed < 0.1

    def test_the_default_gap_respects_the_strictest_service(self) -> None:
        # NCBI allows three requests a second without a key. Asserted against the constant
        # rather than the live limiter, which the fixture above deliberately zeroes.
        assert MIN_SECONDS_BETWEEN_REQUESTS >= 1 / 3
        assert RateLimiter().min_interval == MIN_SECONDS_BETWEEN_REQUESTS


class TestJsonHandling:
    def test_a_body_that_is_not_json_is_reported_clearly(self, transport) -> None:
        transport.responses = [FakeResponse(body=b"<html>down for maintenance</html>")]

        with pytest.raises(WebRequestError, match="did not return JSON"):
            request_json(UNIPROT)

    def test_json_is_requested_explicitly(self, transport) -> None:
        request_json(UNIPROT)

        assert transport.requests[0]["headers"]["Accept"] == "application/json"

    def test_the_library_identifies_itself(self, transport) -> None:
        # Services ask clients to say who they are, and throttle those that do not
        request_text(UNIPROT)

        assert "virtual-lab" in transport.requests[0]["headers"]["User-Agent"]

    @pytest.mark.parametrize("url", ["https://evil.com/x", "http://rest.uniprot.org/x"])
    def test_a_disallowed_url_is_refused_before_any_request(self, transport, url: str) -> None:
        with pytest.raises(DisallowedHostError):
            request_text(url)

        assert transport.requests == []

    def test_a_timeout_is_always_set(self, transport) -> None:
        # Without one, a service that stops responding holds the meeting open indefinitely
        request_text(UNIPROT)

        assert transport.requests[0]["timeout"] > 0

    def test_undecodable_bytes_do_not_raise(self, transport) -> None:
        transport.responses = [FakeResponse(body=b'{"a": "\xff\xfe"}')]

        assert isinstance(request_text(UNIPROT), str)


class TestPostedQueries:
    """post_json is the entry point for search APIs that take a structured query.

    It has no second line of defence: request_text hands off to follow, which re-checks every
    hop, whereas here the check at the entrance is the only one.
    """

    @pytest.fixture
    def post_transport(self, monkeypatch):
        class Transport:
            def __init__(self) -> None:
                self.responses: list[FakeResponse] = []
                self.requests: list[dict] = []

            def post(self, url, **kwargs):
                self.requests.append({"url": url, **kwargs})

                if not self.responses:
                    return FakeResponse(url=url)

                return self.responses.pop(0)

        fake = Transport()
        monkeypatch.setattr(web.requests, "post", fake.post)

        return fake

    def test_a_query_is_sent_and_the_answer_parsed(self, post_transport) -> None:
        post_transport.responses = [FakeResponse(body=b'{"result_set": []}')]

        assert web.post_json(RCSB, payload={"query": {}}) == {"result_set": []}
        assert post_transport.requests[0]["json"] == {"query": {}}

    @pytest.mark.parametrize("url", ["https://evil.com/search", "http://search.rcsb.org/s"])
    def test_a_disallowed_destination_is_refused_before_anything_is_sent(
        self, post_transport, url: str
    ) -> None:
        with pytest.raises(DisallowedHostError):
            web.post_json(url, payload={"query": {}})

        assert post_transport.requests == []

    def test_a_redirect_is_refused_rather_than_replayed(self, post_transport) -> None:
        # A body replayed to a new address is not something to do quietly, and ok() is true for
        # a 3xx, so without an explicit check this reads as a service that "did not return JSON"
        post_transport.responses = [
            FakeResponse(status_code=302, headers={"Location": "https://evil.com/x"})
        ]

        with pytest.raises(WebRequestError, match="not replayed"):
            web.post_json(RCSB, payload={"query": {}})

        assert len(post_transport.requests) == 1

    def test_the_transport_is_told_not_to_follow_redirects(self, post_transport) -> None:
        web.post_json(RCSB, payload={})

        assert post_transport.requests[0]["allow_redirects"] is False

    def test_a_temporary_failure_is_retried(self, post_transport) -> None:
        post_transport.responses = [
            FakeResponse(status_code=503),
            FakeResponse(body=b'{"ok": true}'),
        ]

        assert web.post_json(RCSB, payload={}) == {"ok": True}
        assert len(post_transport.requests) == 2

    def test_a_permanent_failure_is_not_retried(self, post_transport) -> None:
        post_transport.responses = [FakeResponse(status_code=400) for _ in range(3)]

        with pytest.raises(WebRequestError, match="400"):
            web.post_json(RCSB, payload={})

        assert len(post_transport.requests) == 1

    def test_an_oversized_answer_is_refused(self, post_transport) -> None:
        post_transport.responses = [FakeResponse(body=b"x" * 5000)]

        with pytest.raises(ResponseTooLargeError):
            web.post_json(RCSB, payload={}, max_bytes=1000)

    def test_the_rate_limit_applies(self, post_transport, monkeypatch) -> None:
        waited: list[str] = []
        monkeypatch.setattr(web.RATE_LIMITER, "wait", waited.append)

        web.post_json(RCSB, payload={})

        assert waited == ["search.rcsb.org"]

    def test_a_posted_query_is_not_cached(self, post_transport) -> None:
        # A search is not idempotent in the way a lookup by accession is, and the payload is not
        # part of any key here, so two different queries must not be able to share an answer
        post_transport.responses = [FakeResponse(body=b'{"n": 1}'), FakeResponse(body=b'{"n": 2}')]

        assert web.post_json(RCSB, payload={"q": "a"}) == {"n": 1}
        assert web.post_json(RCSB, payload={"q": "b"}) == {"n": 2}


class TestThePubmedToolUsesThisLayer:
    """The PubMed tool is the one network path a model already had, so it goes through here too."""

    def test_the_search_query_is_sent_as_a_parameter(self, transport) -> None:
        # Interpolated into the URL, a query containing & or # would silently alter the request
        transport.responses = [
            FakeResponse(body=b'{"esearchresult": {"idlist": []}}'),
        ]

        from virtual_lab.utils import run_pubmed_search

        run_pubmed_search("spike protein & binding #1", num_articles=1)

        assert transport.requests[0]["params"]["term"] == "spike protein & binding #1"
        assert "spike" not in transport.requests[0]["url"]

    def test_a_query_the_service_cannot_parse_reports_no_articles(self, transport) -> None:
        # NCBI answers an unparseable query with HTTP 200 and an ERROR object where the list of
        # results should be. The query comes from a model, so unbalanced parentheses are routine.
        transport.responses = [
            FakeResponse(body=b'{"esearchresult": {"ERROR": "Empty Term in the request"}}')
        ]

        from virtual_lab.utils import run_pubmed_search

        assert "No articles found" in run_pubmed_search("(((", num_articles=1)

    def test_an_article_that_cannot_be_fetched_is_skipped(self, transport) -> None:
        from virtual_lab.utils import get_pubmed_central_article

        transport.responses = [FakeResponse(status_code=404)]

        assert get_pubmed_central_article("9999999") == (None, None)

    def test_an_article_returning_html_is_skipped(self, transport) -> None:
        from virtual_lab.utils import get_pubmed_central_article

        transport.responses = [FakeResponse(body=b"<html>error</html>")]

        assert get_pubmed_central_article("9999999") == (None, None)

    def test_the_article_identifier_is_encoded(self, transport) -> None:
        from virtual_lab.utils import get_pubmed_central_article

        transport.responses = [FakeResponse(status_code=404)]

        get_pubmed_central_article("../../etc/passwd")

        assert "../.." not in transport.urls[0]


class TestAgainstTheRealServices:
    """Skipped unless VIRTUAL_LAB_LIVE_TESTS=1, since these depend on someone else's uptime."""

    @live_only
    def test_uniprot_answers(self) -> None:
        data = request_json("https://rest.uniprot.org/uniprotkb/P01308.json")

        assert data["primaryAccession"] == "P01308"
        assert data["sequence"]["length"] == 110

    @live_only
    def test_an_unknown_identifier_fails_without_retrying_forever(self) -> None:
        with pytest.raises(WebRequestError):
            request_json(
                "https://rest.uniprot.org/uniprotkb/NOT_AN_ACCESSION_XYZ.json", use_cache=False
            )
