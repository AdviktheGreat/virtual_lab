"""Tests for requests made on behalf of a model.

These run offline. The transport is faked so that redirects, rate limits, oversized bodies, and
retries can be tested deterministically, none of which a real service can be asked to produce on
demand. A handful of tests do hit the real services and are skipped unless VIRTUAL_LAB_LIVE_TESTS
is set, so the default suite stays fast and does not depend on a network or on someone else's
uptime.
"""

import os

import pytest
import requests

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

live_only = pytest.mark.skipif(
    os.environ.get("VIRTUAL_LAB_LIVE_TESTS") != "1",
    reason="Set VIRTUAL_LAB_LIVE_TESTS=1 to query the real services",
)

UNIPROT = "https://rest.uniprot.org/uniprotkb/P01308.json"


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
        self.headers = headers or {}
        self.url = url
        self.encoding = "utf-8"

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

    @pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS))
    def test_every_listed_host_is_accepted(self, host: str) -> None:
        assert check_host(f"https://{host}/path") == f"https://{host}/path"

    def test_a_listed_host_is_matched_case_insensitively(self) -> None:
        assert check_host("https://REST.UniProt.ORG/x")

    def test_the_error_names_the_hosts_that_are_allowed(self) -> None:
        with pytest.raises(DisallowedHostError, match="rest.uniprot.org"):
            check_host("https://evil.com/x")

    def test_the_list_is_only_scientific_databases(self) -> None:
        # A general purpose host on this list would turn every tool into an open proxy
        assert all(
            host.endswith((".gov", ".org", ".uk", ".edu")) for host in ALLOWED_HOSTS
        ), sorted(ALLOWED_HOSTS)


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

    def test_caching_can_be_declined(self, transport) -> None:
        transport.responses = [FakeResponse(body=b"{}"), FakeResponse(body=b"{}")]

        request_json(UNIPROT, use_cache=False)
        request_json(UNIPROT, use_cache=False)

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

    def test_a_timeout_is_always_set(self, transport) -> None:
        # Without one, a service that stops responding holds the meeting open indefinitely
        request_text(UNIPROT)

        assert transport.requests[0]["timeout"] > 0

    def test_undecodable_bytes_do_not_raise(self, transport) -> None:
        transport.responses = [FakeResponse(body=b'{"a": "\xff\xfe"}')]

        assert isinstance(request_text(UNIPROT), str)


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
