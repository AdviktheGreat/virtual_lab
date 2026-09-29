"""Tests for the literature searches agents use.

Offline against fixtures trimmed from real responses, with live tests at the end that check the
fixtures still describe reality.

The thing being defended against here is not a lookup that fails. It is a search that succeeds
and answers a different question than the one asked: arXiv reads three loose words as
alternatives and returns 1,285,358 results, Europe PMC reads the same three as requirements and
returns 922, and both report a confident count either way. Most of what follows is about the
difference between the query a caller wrote and the query a service ran.
"""

import os
import time
import xml.etree.ElementTree as ElementTree

import pytest

from conftest import FakeResponse
from virtual_lab.constants import (
    MAX_ABSTRACT_CHARACTERS,
    MAX_ARTICLE_CHARACTERS,
    MAX_AUTHORS_REPORTED,
    MAX_ITEMS_LISTED,
    MAX_SEARCH_RESULTS,
    MAX_SECTION_CHARACTERS,
    WEB_MAX_ATTEMPTS,
)
from virtual_lab.literature import (
    ATOM,
    Article,
    ArticleText,
    Preprint,
    PreprintResults,
    article_from,
    arxiv_query,
    collapse,
    get_article,
    get_article_text,
    preprint_from,
    search_articles,
    search_preprints,
    section_text,
    yes,
)
from virtual_lab.records import RecordNotFoundError, parse_xml
from virtual_lab.web import WebRequestError

live_only = pytest.mark.skipif(
    os.environ.get("VIRTUAL_LAB_LIVE_TESTS") != "1",
    reason="Set VIRTUAL_LAB_LIVE_TESTS=1 to query the real services",
)


# Trimmed from a real Europe PMC lite search result. Note the flags: eleven of the fields on a
# real record are the letter Y or the letter N, and pubYear is a string while citedByCount is
# a number, in the same record.
ARTICLE_RECORD = {
    "id": "21937511",
    "source": "MED",
    "pmid": "21937511",
    "pmcid": "PMC3258128",
    "doi": "10.1093/nar/gkr715",
    "title": "Hepato-specific microRNA-122 facilitates accumulation of newly synthesized miRNA.",
    "authorString": "Li S, Zhu J, Fu H, Wan J.",
    "journalTitle": "Nucleic acids research",
    "pubYear": "2012",
    "citedByCount": 29,
    "isOpenAccess": "Y",
    "inEPMC": "Y",
    "abstractText": "microRNAs are a versatile class of non-coding RNAs.",
}

# A preprint, which has no journal and records its server under the book details instead
PREPRINT_RECORD = {
    "id": "PPR1318417",
    "source": "PPR",
    "doi": "10.64898/2026.09.10.750412",
    "title": "Agentic-AI-ready genome-wide poxvirus-host interaction screen",
    "authorString": "Brown A, Green B.",
    "pubYear": "2026",
    "citedByCount": 0,
    "isOpenAccess": "N",
    "inEPMC": "N",
    "bookOrReportDetails": {"publisher": "bioRxiv", "yearOfPublication": 2026},
}

ARXIV_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>893</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2410.08355v3</id>
    <title>Metalic: Meta-Learning In-Context
  with Protein Language Models</title>
    <summary>Predicting the biophysical properties of proteins.</summary>
    <published>2024-10-10T20:19:35Z</published>
    <updated>2025-07-15T22:56:19Z</updated>
    <author><name>Jacob Beck</name></author>
    <author><name>Shikha Surana</name></author>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="q-bio.BM" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:comment>Published at ICLR 2025</arxiv:comment>
    <arxiv:journal_ref>The Thirteenth ICLR</arxiv:journal_ref>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>
"""

# One section of each kind a full text has: the abstract, a section worth returning with a
# subsection inside it, and a section that is a list of other people's papers
JATS = """<?xml version="1.0"?>
<article>
  <front><article-meta>
    <abstract><title>Abstract</title><p>What the paper found.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>INTRODUCTION</title><p>Why this matters, as shown before <xref>1</xref>.</p></sec>
    <sec><title>RESULTS</title>
      <sec><title>Affinity purification</title><p>The first thing measured.</p></sec>
      <sec><title>Binding</title><p>The second thing measured.</p></sec>
    </sec>
    <sec><title>FUNDING</title><p>Grant 12345.</p></sec>
    <sec><title>REFERENCES</title><p>Someone else. A different paper entirely. 2004.</p></sec>
  </body>
</article>
"""


def search_response(*records, hit_count=None):
    """Builds a Europe PMC search response around some records."""
    return {
        "hitCount": len(records) if hit_count is None else hit_count,
        "resultList": {"result": list(records)},
    }


def queue(transport, *bodies):
    """Queues JSON responses in the order the code will ask for them."""
    transport.responses = [FakeResponse(json_body=body) for body in bodies]


def queue_text(transport, *bodies):
    """Queues text responses in the order the code will ask for them."""
    transport.responses = [FakeResponse(body=body.encode()) for body in bodies]


class TestFlagsSentAsLetters:
    """Europe PMC sends its booleans as the letter Y or the letter N."""

    def test_the_letter_n_is_not_true(self) -> None:
        # "N" is a non-empty string, so every one of these fields is true when tested directly
        assert yes("N") is False

    @pytest.mark.parametrize("value, expected", [("Y", True), ("y", True), ("N", False),
                                                 ("", False), (None, False)])
    def test_a_flag_is_read_as_what_it_says(self, value, expected) -> None:
        assert yes(value) is expected

    def test_an_article_that_is_not_open_access_does_not_claim_to_be(self) -> None:
        article = article_from(ARTICLE_RECORD | {"isOpenAccess": "N"})

        assert article.open_access is False
        assert article.has_full_text is False


class TestReadingAnArticleRecord:
    def test_the_fields_of_an_article_are_parsed(self) -> None:
        article = article_from(ARTICLE_RECORD)

        assert article.pmid == "21937511"
        assert article.pmcid == "PMC3258128"
        assert article.journal == "Nucleic acids research"
        assert article.cited_by == 29
        assert article.open_access is True

    def test_a_year_sent_as_a_string_becomes_a_number(self) -> None:
        # pubYear arrives as "2012" while citedByCount beside it arrives as a number
        assert article_from(ARTICLE_RECORD).year == 2012
        assert article_from(ARTICLE_RECORD | {"pubYear": "n/a"}).year is None

    def test_the_author_string_does_not_end_up_with_a_full_stop_in_a_name(self) -> None:
        # Europe PMC's authorString ends in a full stop, which becomes part of the last name
        assert article_from(ARTICLE_RECORD).authors == ("Li S", "Zhu J", "Fu H", "Wan J")

    def test_a_structured_author_list_is_preferred_to_the_rendered_string(self) -> None:
        article = article_from(
            ARTICLE_RECORD | {"authorList": {"author": [{"fullName": "Ada Lovelace"}]}}
        )

        assert article.authors == ("Ada Lovelace",)

    def test_a_preprint_reports_its_server_rather_than_no_venue(self) -> None:
        article = article_from(PREPRINT_RECORD)

        assert article.is_preprint is True
        assert article.journal == "bioRxiv"
        assert "preprint" in article.summary().lower()

    def test_a_preprint_report_says_it_was_not_peer_reviewed(self) -> None:
        assert "not been peer reviewed" in article_from(PREPRINT_RECORD).report()

    def test_a_published_article_is_not_labelled_a_preprint(self) -> None:
        assert "peer reviewed" not in article_from(ARTICLE_RECORD).report()

    def test_an_author_list_too_long_to_print_is_counted_instead(self) -> None:
        many = tuple(f"Author {index}" for index in range(30))
        article = Article("1", authors=many)

        assert f"Author {MAX_AUTHORS_REPORTED - 1}" in article.credit()
        assert f"Author {MAX_AUTHORS_REPORTED}," not in article.credit()
        assert f"and {30 - MAX_AUTHORS_REPORTED} others" in article.credit()

    def test_a_long_abstract_is_capped(self) -> None:
        article = article_from(ARTICLE_RECORD | {"abstractText": "w " * 4_000})
        report = article.report()

        assert "characters not shown" in report
        assert len(report) < MAX_ABSTRACT_CHARACTERS + 1_000

    def test_an_article_with_no_abstract_says_so(self) -> None:
        assert "No abstract" in article_from(ARTICLE_RECORD | {"abstractText": ""}).report()


class TestWhetherAFullTextExists:
    """Both conditions are needed, and Europe PMC does not say so in one field."""

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({}, True),
            ({"pmcid": ""}, False),
            ({"isOpenAccess": "N"}, False),
            ({"inEPMC": "N"}, False),
        ],
    )
    def test_an_article_is_readable_only_when_every_part_is_there(
        self, overrides, expected
    ) -> None:
        # An article can be open access elsewhere on the web while Europe PMC holds only the
        # abstract, and asking for the text of one of those is answered with 500
        assert article_from(ARTICLE_RECORD | overrides).has_full_text is expected

    def test_the_report_says_whether_the_text_can_be_had(self) -> None:
        assert "Full text can be fetched" in article_from(ARTICLE_RECORD).report()
        assert "does not hold the full text" in article_from(
            ARTICLE_RECORD | {"inEPMC": "N"}
        ).report()


class TestSearchingEuropePmc:
    def test_the_search_reaches_europe_pmc(self, web_transport) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122")

        assert "europepmc/webservices/rest/search" in web_transport.urls[0]
        assert web_transport.params[0]["query"] == "miR-122"

    def test_the_cheaper_result_type_is_asked_for(self, web_transport) -> None:
        # A core result is nine times the size of a lite one, and a listing does not show the
        # fields that make up the difference
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122")

        assert web_transport.params[0]["resultType"] == "lite"

    @pytest.mark.parametrize(
        "sort, expected", [("relevance", None), ("cited", "CITED desc"), ("recent", "P_PDATE_D desc")]
    )
    def test_each_ordering_is_passed_through(self, web_transport, sort, expected) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122", sort=sort)

        assert web_transport.params[0].get("sort") == expected

    def test_an_unknown_ordering_is_refused_before_any_request(self, web_transport) -> None:
        with pytest.raises(ValueError, match="Unknown sort"):
            search_articles("miR-122", sort="best")

        assert web_transport.requests == []

    def test_a_search_for_nothing_is_refused(self, web_transport) -> None:
        with pytest.raises(ValueError, match="something to search for"):
            search_articles("   ")

        assert web_transport.requests == []

    def test_the_filters_are_applied_by_the_service(self, web_transport) -> None:
        # Filtering the results here instead would return fewer than the caller asked for and
        # give no reason why
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122", open_access_only=True, include_preprints=False)

        sent = web_transport.params[0]["query"]

        assert 'OPEN_ACCESS:"Y"' in sent
        assert 'IN_EPMC:"Y"' in sent
        assert 'NOT SRC:"PPR"' in sent

    def test_the_query_is_bracketed_before_a_filter_is_joined_to_it(self, web_transport) -> None:
        # AND binds tighter than OR, so "a OR b" with a filter appended is read as
        # "a OR (b AND filter)" and returns everything matching a, filtered or not
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("nanobody OR VHH", open_access_only=True)

        assert web_transport.params[0]["query"].startswith("(nanobody OR VHH) AND ")

    def test_an_unfiltered_query_is_not_bracketed_for_no_reason(self, web_transport) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("nanobody OR VHH")

        assert web_transport.params[0]["query"] == "nanobody OR VHH"

    def test_the_filters_are_left_off_when_not_asked_for(self, web_transport) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122")

        assert web_transport.params[0]["query"] == "miR-122"

    def test_a_search_is_held_to_what_is_worth_requesting(self, web_transport) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD))
        search_articles("miR-122", limit=500)

        assert web_transport.params[0]["pageSize"] == MAX_SEARCH_RESULTS

    def test_a_service_that_ignores_the_limit_is_still_held_to_it(self, web_transport) -> None:
        queue(web_transport, search_response(*([ARTICLE_RECORD] * 40), hit_count=40))

        assert len(search_articles("miR-122", limit=3).articles) == 3

    def test_the_results_are_rendered_with_what_a_choice_needs(self, web_transport) -> None:
        queue(web_transport, search_response(ARTICLE_RECORD, hit_count=922))
        report = search_articles("miR-122", limit=1).report()

        assert "922 articles" in report
        assert "Nucleic acids research" in report
        assert "cited 29" in report
        assert "full text: PMC3258128" in report

    def test_a_search_that_found_nothing_explains_why_it_might_have(self, web_transport) -> None:
        # The two services read a loose query in opposite ways, and a caller who gets nothing
        # from one has no reason to guess that is why
        queue(web_transport, search_response())
        report = search_articles("a very long and specific query").report()

        assert "No articles" in report
        assert "must all appear" in report
        assert "arxiv_search" in report


class TestLookingUpOneArticle:
    @pytest.mark.parametrize(
        "identifier, expected",
        [
            ("PMC3258128", "PMCID:PMC3258128"),
            ("pmc3258128", "PMCID:PMC3258128"),
            ("21937511", "EXT_ID:21937511 AND SRC:MED"),
            ("10.1093/nar/gkr715", 'DOI:"10.1093/nar/gkr715"'),
        ],
    )
    def test_each_kind_of_identifier_is_asked_for_the_way_it_has_to_be(
        self, web_transport, identifier, expected
    ) -> None:
        # Not a style choice. PMCID:"PMC3258128" and EXT_ID:"21937511" both match nothing at
        # all, while the same queries unquoted match the article; a DOI needs the quotes.
        queue(web_transport, search_response(ARTICLE_RECORD))
        get_article(identifier)

        assert web_transport.params[0]["query"] == expected

    def test_the_fuller_result_type_is_asked_for(self, web_transport) -> None:
        # This is the one place the abstract is wanted, which a lite result does not carry
        queue(web_transport, search_response(ARTICLE_RECORD))
        get_article("PMC3258128")

        assert web_transport.params[0]["resultType"] == "core"

    def test_an_identifier_that_could_carry_query_syntax_is_refused(self, web_transport) -> None:
        # The two unquoted forms are checked against their shape, so only a DOI reaches the
        # quoted branch, and a quote inside one would end the quoting early
        with pytest.raises(ValueError, match="not a PMID"):
            get_article('10.1/x" OR DOI:"10.2/y')

        assert web_transport.requests == []

    def test_an_empty_identifier_is_refused_before_any_request(self, web_transport) -> None:
        with pytest.raises(ValueError, match="PMID"):
            get_article("  ")

        assert web_transport.requests == []

    def test_an_identifier_that_matches_nothing_is_reported_as_missing(self, web_transport) -> None:
        queue(web_transport, search_response())

        with pytest.raises(RecordNotFoundError, match="PMC9999999"):
            get_article("PMC9999999")

    def test_an_outage_is_not_reported_as_a_missing_article(self, web_transport) -> None:
        web_transport.responses = [FakeResponse(status_code=503) for _ in range(WEB_MAX_ATTEMPTS)]

        with pytest.raises(Exception) as raised:
            get_article("PMC3258128")

        assert isinstance(raised.value, WebRequestError)
        assert not isinstance(raised.value, RecordNotFoundError)


class TestReadingAFullText:
    def test_the_record_is_checked_before_the_text_is_requested(self, web_transport) -> None:
        # Europe PMC answers a request for the text of an article it does not hold with 500,
        # the same status an outage gives, so three attempts and several seconds of backoff are
        # spent before it reports that the service could not be reached
        queue(web_transport, search_response(ARTICLE_RECORD | {"inEPMC": "N"}))

        with pytest.raises(RecordNotFoundError, match="no open full text"):
            get_article_text("PMC3258128")

        assert len(web_transport.requests) == 1

    def test_the_text_is_fetched_once_the_record_says_it_exists(self, web_transport) -> None:
        web_transport.responses = [
            FakeResponse(json_body=search_response(ARTICLE_RECORD)),
            FakeResponse(body=JATS.encode()),
        ]
        get_article_text("PMC3258128")

        assert web_transport.urls[1].endswith("/PMC3258128/fullTextXML")

    def test_an_empty_identifier_is_refused_before_any_request(self, web_transport) -> None:
        with pytest.raises(ValueError, match="PMCID"):
            get_article_text("  ")

        assert web_transport.requests == []

    def read(self, web_transport, xml=JATS, record=None):
        """Fetches a full text against a fixture."""
        web_transport.responses = [
            FakeResponse(json_body=search_response(record or ARTICLE_RECORD)),
            FakeResponse(body=xml.encode()),
        ]

        return get_article_text("PMC3258128")

    def test_the_sections_worth_reading_are_kept(self, web_transport) -> None:
        text = self.read(web_transport)

        assert [heading for heading, _ in text.sections] == ["Abstract", "INTRODUCTION", "RESULTS"]

    def test_the_sections_nobody_asked_for_are_left_out_but_named(self, web_transport) -> None:
        # References are 28% of the body text of a typical article and are a list of other
        # papers' titles, which reads to a model as though this article discussed all of them
        text = self.read(web_transport)

        assert text.skipped == ("FUNDING", "REFERENCES")
        assert "A different paper entirely" not in text.report()
        assert "Sections not shown: FUNDING, REFERENCES" in text.report()

    def test_a_heading_does_not_run_into_the_text_it_names(self, web_transport) -> None:
        # itertext() reaches the title first, so the abstract otherwise opens
        # "AbstractWhat the paper found" under a heading that already said Abstract
        text = self.read(web_transport)

        assert dict(text.sections)["Abstract"] == "What the paper found."

    def test_a_subsection_keeps_its_own_heading_and_is_marked_off_from_it(
        self, web_transport
    ) -> None:
        results = dict(self.read(web_transport).sections)["RESULTS"]

        assert results == "Affinity purification. The first thing measured. Binding. The second thing measured."

    def test_a_long_article_is_cut_to_what_a_meeting_can_hold(self, web_transport) -> None:
        # Enough sections, each long enough, that the whole-article budget runs out before the
        # sections do. Inflating one section only reaches the per-section cap, which is a
        # different limit: the article budget would go untested and its loss unnoticed.
        sections = "".join(
            f"<sec><title>Section {index}</title><p>{'word ' * 2_000}</p></sec>"
            for index in range(20)
        )
        big = JATS.replace("<sec><title>INTRODUCTION</title>", sections + "<sec><title>X</title>")
        report = self.read(web_transport, xml=big).report()

        assert len(report) < MAX_ARTICLE_CHARACTERS + 2_000
        assert "The rest of the article is not shown" in report
        assert "Section 19" not in report

    def test_one_long_section_does_not_use_the_whole_budget(self, web_transport) -> None:
        big = JATS.replace("Why this matters, as shown before ", "word " * 20_000)
        text = self.read(web_transport, xml=big)
        report = text.report()

        # The section is cut to its own limit, so what comes after it still appears
        assert "RESULTS" in report
        assert report.index("## INTRODUCTION") < report.index("## RESULTS")

        introduction = dict(text.sections)["INTRODUCTION"]

        # Bounded against the section limit rather than against the article one, which is five
        # times larger and would hold whatever this section grew to
        assert len(introduction) > MAX_SECTION_CHARACTERS
        assert report.index("## RESULTS") < MAX_SECTION_CHARACTERS + 1_000


class TestRefusingAnUnsafeDocument:
    def test_a_document_declaring_entities_is_refused(self) -> None:
        # ElementTree resolves internal entities, which was checked rather than assumed: a
        # document declaring a few nested ones expands to whatever size its author chose
        bomb = (
            '<?xml version="1.0"?><!DOCTYPE f [<!ENTITY a "AAAAAAAAAA">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
            "<feed><entry><title>&b;</title></entry></feed>"
        )

        with pytest.raises(WebRequestError, match="document type"):
            parse_xml(bomb, WebRequestError)

    def test_the_check_does_not_depend_on_how_the_declaration_is_written(self) -> None:
        with pytest.raises(WebRequestError, match="document type"):
            parse_xml('<?xml version="1.0"?>\n  <!doctype feed []>\n<feed/>', WebRequestError)

    def test_a_declaration_after_a_long_comment_is_still_refused(self) -> None:
        # Only the first 4 KB used to be searched, so a comment in front of the declaration let a
        # 6.6 KB feed through that expanded to a five megabyte report
        body = (
            '<?xml version="1.0"?><!--' + "padding " * 1000 + '--><!DOCTYPE f [<!ENTITY a "b">]>'
            "<feed><entry><title>&a;</title></entry></feed>"
        )

        with pytest.raises(WebRequestError, match="document type"):
            parse_xml(body, WebRequestError)

    def test_an_entity_declaration_without_a_doctype_is_refused(self) -> None:
        with pytest.raises(WebRequestError, match="document type"):
            parse_xml('<feed><!entity a "b"></feed>', WebRequestError)

    def test_an_ordinary_document_is_parsed(self) -> None:
        assert parse_xml("<feed><entry/></feed>", WebRequestError).tag == "feed"

    def test_something_that_is_not_xml_is_a_request_error(self) -> None:
        # Not a ParseError escaping, which would bypass every caller's handling
        with pytest.raises(WebRequestError, match="not usable XML"):
            parse_xml("<html>a service error page</html>>", WebRequestError)


class TestWhatArxivIsActuallyAsked:
    """arXiv reads loose words as alternatives, which is the opposite of what anyone means."""

    def test_loose_words_become_terms_that_must_all_appear(self) -> None:
        # "protein language model" unchanged returns 1,285,358 results, being everything
        # mentioning any of the three words; requiring all three returns 893
        assert arxiv_query("protein language model") == (
            "all:protein AND all:language AND all:model"
        )

    def test_a_single_word_is_still_given_a_field(self) -> None:
        assert arxiv_query("nanobody") == "all:nanobody"

    def test_a_quoted_phrase_stays_one_term(self) -> None:
        assert arxiv_query('"protein language model"') == 'all:"protein language model"'

    def test_a_phrase_beside_a_word_keeps_both(self) -> None:
        assert arxiv_query('"language model" protein') == (
            'all:"language model" AND all:protein'
        )

    @pytest.mark.parametrize(
        "query",
        [
            "all:protein AND cat:q-bio.BM",
            "ti:transformer",
            "au:Bonvin",
            "protein AND folding",
        ],
    )
    def test_a_query_that_knows_the_syntax_is_left_alone(self, query) -> None:
        assert arxiv_query(query) == query

    def test_a_word_that_merely_contains_a_field_name_is_not_one(self) -> None:
        # "id:" is a field, and as a substring it matches "covid:", which would hand a plain
        # query through unchanged and turn it back into an OR
        assert arxiv_query("covid: vaccine") == "all:covid: AND all:vaccine"

    def test_the_report_says_what_was_actually_searched(self) -> None:
        results = PreprintResults(
            query="all:protein AND all:language",
            asked="protein language",
            total=893,
            preprints=(Preprint("1", title="A paper"),),
        )
        report = results.report()

        assert "Searched as all:protein AND all:language" in report
        assert '"protein language"' in report

    def test_nothing_is_said_when_the_query_went_as_written(self) -> None:
        results = PreprintResults(
            query="ti:transformer",
            asked="ti:transformer",
            total=1,
            preprints=(Preprint("1", title="A paper"),),
        )

        assert "Searched as" not in results.report()


class TestSearchingArxiv:
    def test_the_search_reaches_arxiv(self, web_transport) -> None:
        queue_text(web_transport, ARXIV_FEED)
        search_preprints("protein language model")

        assert "export.arxiv.org/api/query" in web_transport.urls[0]
        assert web_transport.params[0]["search_query"] == (
            "all:protein AND all:language AND all:model"
        )

    def test_a_category_narrows_the_query_without_breaking_it(self, web_transport) -> None:
        # Bracketed, since the terms are joined with AND and an unbracketed category would bind
        # to the last of them only
        queue_text(web_transport, ARXIV_FEED)
        search_preprints("protein language model", category="q-bio.BM")

        assert web_transport.params[0]["search_query"] == (
            "(all:protein AND all:language AND all:model) AND cat:q-bio.BM"
        )

    def test_a_search_for_nothing_is_refused(self, web_transport) -> None:
        with pytest.raises(ValueError, match="something to search for"):
            search_preprints("  ")

        assert web_transport.requests == []

    @pytest.mark.parametrize(
        "sort, expected",
        [
            ("relevance", "relevance"),
            ("recent", "submittedDate"),
            ("updated", "lastUpdatedDate"),
        ],
    )
    def test_each_ordering_is_passed_through(self, web_transport, sort, expected) -> None:
        queue_text(web_transport, ARXIV_FEED)
        search_preprints("nanobody", sort=sort)

        assert web_transport.params[0]["sortBy"] == expected
        assert web_transport.params[0]["sortOrder"] == "descending"

    def test_an_unknown_ordering_is_refused_before_any_request(self, web_transport) -> None:
        with pytest.raises(ValueError, match="Unknown sort"):
            search_preprints("nanobody", sort="cited")

        assert web_transport.requests == []

    def test_a_search_is_held_to_what_is_worth_requesting(self, web_transport) -> None:
        queue_text(web_transport, ARXIV_FEED)
        search_preprints("nanobody", limit=500)

        assert web_transport.params[0]["max_results"] == MAX_SEARCH_RESULTS

    def test_the_count_comes_from_the_feed(self, web_transport) -> None:
        queue_text(web_transport, ARXIV_FEED)

        assert search_preprints("protein language model").total == 893

    def test_a_feed_with_no_count_falls_back_to_what_it_holds(self, web_transport) -> None:
        queue_text(web_transport, ARXIV_FEED.replace("893", "not a number"))

        assert search_preprints("protein language model").total == 1

    def test_a_service_that_ignores_the_limit_is_still_held_to_it(self, web_transport) -> None:
        # max_results is asked for and trusted nowhere: the feed is cut on arrival too
        one_entry = ARXIV_FEED[ARXIV_FEED.index("<entry>") : ARXIV_FEED.index("</entry>") + 8]
        crowded = ARXIV_FEED.replace(one_entry, one_entry * 40)
        queue_text(web_transport, crowded)

        assert len(search_preprints("nanobody", limit=3).preprints) == 3


class TestReadingAPreprintRecord:
    def entry(self, feed=ARXIV_FEED):
        """Parses the one entry in a feed."""
        root = parse_xml(feed, WebRequestError)

        return preprint_from(root.find("{http://www.w3.org/2005/Atom}entry"))

    def test_the_fields_of_a_preprint_are_parsed(self) -> None:
        preprint = self.entry()

        assert preprint.arxiv_id == "2410.08355v3"
        assert preprint.authors == ("Jacob Beck", "Shikha Surana")
        assert preprint.categories == ("cs.LG", "q-bio.BM")
        assert preprint.url == "https://arxiv.org/abs/2410.08355v3"

    def test_the_version_is_kept(self) -> None:
        # A result for v1 of a paper now on v3 is a different document
        assert self.entry().arxiv_id.endswith("v3")

    def test_a_title_wrapped_across_lines_is_put_back_on_one(self) -> None:
        assert self.entry().title == "Metalic: Meta-Learning In-Context with Protein Language Models"

    def test_a_category_listed_twice_is_named_once(self) -> None:
        twice = ARXIV_FEED.replace('term="q-bio.BM"', 'term="cs.LG"')

        assert self.entry(twice).categories == ("cs.LG",)

    def test_a_preprint_says_it_is_one(self) -> None:
        assert "preprint" in self.entry().report().lower()

    def test_where_it_was_published_is_preferred_to_the_authors_note(self) -> None:
        # The note is where "accepted at NeurIPS" lives, but a journal reference is the stronger
        # claim and printing both says the same thing twice
        report = self.entry().report()

        assert "The Thirteenth ICLR" in report
        assert "Published at ICLR 2025" not in report

    def test_the_note_is_shown_when_there_is_no_journal_reference(self) -> None:
        without = ARXIV_FEED.replace(
            "<arxiv:journal_ref>The Thirteenth ICLR</arxiv:journal_ref>", ""
        )

        assert "Published at ICLR 2025" in self.entry(without).report()

    def test_a_revision_date_is_shown_only_when_it_differs(self) -> None:
        assert "Revised 2025-07-15" in self.entry().report()

        same = ARXIV_FEED.replace("2025-07-15T22:56:19Z", "2024-10-10T20:19:35Z")

        assert "Revised" not in self.entry(same).report()

    def test_a_long_abstract_is_capped(self) -> None:
        long = ARXIV_FEED.replace("Predicting the biophysical properties of proteins.", "w " * 4_000)

        assert "characters not shown" in self.entry(long).report()


class TestCollapsingWhitespace:
    @pytest.mark.parametrize(
        "text, expected",
        [("a\n  b", "a b"), ("  a  ", "a"), ("", ""), ("a\tb", "a b")],
    )
    def test_runs_of_whitespace_become_one_space(self, text, expected) -> None:
        assert collapse(text) == expected


class TestWhatAReportCosts:
    def test_a_full_search_report_stays_small(self, web_transport) -> None:
        queue(
            web_transport,
            search_response(*([ARTICLE_RECORD] * MAX_SEARCH_RESULTS), hit_count=9_000),
        )
        report = search_articles("miR-122", limit=MAX_SEARCH_RESULTS).report()

        assert len(report) < 8_000

    def test_a_search_report_does_not_carry_the_abstracts(self) -> None:
        # The listing is for choosing what to read, and a lite search does not return them
        assert "versatile class" not in article_from(ARTICLE_RECORD).summary()

    def test_a_full_text_report_stays_within_its_budget(self) -> None:
        sections = tuple((f"Section {index}", "word " * 5_000) for index in range(20))
        text = ArticleText(article=article_from(ARTICLE_RECORD), sections=sections)

        assert len(text.report()) < MAX_ARTICLE_CHARACTERS + 2_000


class TestAResponseOfTheWrongShape:
    """A field of the wrong type is read as absent, and a long one is shortened when it is read."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"pubYear": "\u00b2"},
            {"pubYear": "2012a", "citedByCount": "many"},
            {"citedByCount": float("inf")},
            {"authorList": {"author": ["not an object"]}, "authorString": None},
            {"authorList": ["not an object"], "journalInfo": {"journal": "Nature"}},
            {"title": {"text": "A title"}, "bookOrReportDetails": ["bioRxiv"]},
        ],
    )
    def test_an_article_of_the_wrong_shape_still_reports(self, overrides) -> None:
        assert article_from(ARTICLE_RECORD | overrides).report()

    def test_a_year_sent_as_a_number_is_read(self) -> None:
        assert article_from(ARTICLE_RECORD | {"pubYear": 2012}).year == 2012

    def test_every_field_of_an_article_report_is_bounded(self) -> None:
        huge = "x" * 1_000_000
        article = article_from(
            ARTICLE_RECORD
            | {
                "title": huge,
                "journalTitle": huge,
                "authorString": ", ".join([huge] * 3),
                "doi": huge,
                "abstractText": huge,
            }
        )

        assert len(article.report()) < 10_000
        assert len(article.summary()) < 5_000

    @pytest.mark.parametrize(
        "response",
        [
            {"hitCount": "many", "resultList": {"result": "none"}},
            {"hitCount": 3, "resultList": ["not an object"]},
            {"hitCount": 1, "resultList": {"result": ["not an object"]}},
        ],
    )
    def test_a_search_of_the_wrong_shape_still_reports(self, web_transport, response) -> None:
        queue(web_transport, response)

        assert search_articles("microRNA").report()

    def test_a_total_that_is_not_a_number_is_the_number_of_preprints(self, web_transport) -> None:
        queue_text(web_transport, ARXIV_FEED.replace(">893<", ">\u00b2<"))

        assert search_preprints("protein language model").total == 1

    def test_every_field_of_a_preprint_report_is_bounded(self) -> None:
        huge = "x" * 1_000_000
        entry = ElementTree.fromstring(
            ARXIV_FEED.replace("Published at ICLR 2025", huge)
            .replace("The Thirteenth ICLR", huge)
            .replace("Jacob Beck", huge)
            .replace('term="cs.LG"', f'term="{huge}"')
            .replace("<arxiv:primary", f"<arxiv:doi>{huge}</arxiv:doi><arxiv:primary")
            .replace("Metalic", huge)
            .encode()
        ).find("atom:entry", ATOM)
        preprint = preprint_from(entry)

        assert len(preprint.report()) < 10_000
        assert len(preprint.summary()) < 10_000

    def test_the_sections_left_out_are_counted_past_the_first_few(self) -> None:
        text = ArticleText(article=Article(article_id="1"), skipped=("References",) * 100_000)

        assert f"and {100_000 - MAX_ITEMS_LISTED:,} more" in text.report()
        assert len(text.report()) < 5_000

    def test_a_long_section_heading_is_shortened(self, web_transport) -> None:
        huge = "H" * 1_000_000
        queue(web_transport, search_response(ARTICLE_RECORD))
        web_transport.responses.append(
            FakeResponse(body=JATS.replace("INTRODUCTION", huge).encode())
        )

        text = get_article_text("PMC3258128")

        assert len(text.report()) < MAX_ARTICLE_CHARACTERS + 5_000


class TestReadingDeeplyNestedSections:
    """A full text is a document from elsewhere, so how deep its sections go is not ours to pick."""

    def test_sections_nested_past_the_recursion_limit_are_read(self) -> None:
        body = "<sec>" * 2000 + "deepest" + "</sec>" * 2000

        assert section_text(ElementTree.fromstring(body)) == "deepest"

    def test_markup_nested_past_the_recursion_limit_is_read(self) -> None:
        # Read with itertext(), which is safe at this depth only because the C implementation of
        # ElementTree walks it without recursing
        body = "<sec><title>T</title>" + "<p>" * 2000 + "deepest" + "</p>" * 2000 + "</sec>"

        assert section_text(ElementTree.fromstring(body)) == "deepest"

    def test_deep_nesting_costs_time_in_proportion_to_its_size(self) -> None:
        # Collapsing each subsection again inside its parent took 5.85 s for this document
        body = (
            "".join(f"<sec><title>T{index}</title>" + "word " * 200 for index in range(800))
            + "</sec>" * 800
        )
        root = ElementTree.fromstring(body)

        started = time.perf_counter()
        text = section_text(root)

        assert time.perf_counter() - started < 1.0
        assert text.startswith("word word")
        assert text.endswith("T799. " + " ".join(["word"] * 200))

    def test_text_between_and_after_subsections_keeps_its_place(self) -> None:
        body = (
            "<sec><title>Own</title>lead <b>bold</b> tail"
            "<sec><title>Sub</title>inside<sec>deeper</sec>after deeper</sec>"
            " between <fig><title>Figure</title><sec>boxed</sec></fig> end</sec>"
        )

        # Exactly what the recursive version produced, including the two places where it joins
        # words without a space: after an untitled subsection, and inside markup read whole
        assert section_text(ElementTree.fromstring(body)) == (
            "lead bold tail Sub. inside deeperafter deeper between Figureboxed end"
        )


class TestWhatAPreprintReportSays:
    def test_a_long_author_list_is_named_in_part_and_counted(self) -> None:
        authors = tuple(f"Author {index}" for index in range(MAX_AUTHORS_REPORTED + 3))
        preprint = Preprint(arxiv_id="2410.08355v3", authors=authors, published="2024-10-10T20")

        assert preprint.credit().endswith("Author 7, and 3 others. 2024-10-10")

    def test_a_journal_reference_is_marked_in_a_search_result(self) -> None:
        preprint = Preprint(arxiv_id="1", journal_reference="Nature 1", comment="10 pages")

        assert "[published: Nature 1]" in preprint.summary()
        assert "10 pages" not in preprint.summary()

    def test_an_author_note_is_marked_when_there_is_no_journal_reference(self) -> None:
        preprint = Preprint(arxiv_id="1", categories=("cs.LG",), comment="Accepted at  ICLR")

        assert "[cs.LG; Accepted at ICLR]" in preprint.summary()

    def test_a_doi_is_given_its_own_line(self) -> None:
        report = Preprint(arxiv_id="1", doi="10.1000/xyz").report()

        assert "\nDOI: 10.1000/xyz\n" in report

    def test_a_search_with_no_preprints_says_so(self) -> None:
        results = PreprintResults(query="all:x", asked="x", total=0)

        assert results.report() == 'No preprints on arXiv match "x".'


class TestAgainstTheRealServices:
    """These ask the real services whether the fixtures above still describe them."""

    @live_only
    def test_europe_pmc_still_answers_a_search(self) -> None:
        results = search_articles("nanobody SARS-CoV-2 neutralization", limit=3, sort="cited")

        assert results.total > 100
        assert len(results.articles) == 3
        assert all(article.title for article in results.articles)

    @live_only
    def test_europe_pmc_still_sends_its_flags_as_letters(self) -> None:
        # The whole reason yes() exists. If these become real booleans it still works, but the
        # comment explaining it would be wrong.
        from virtual_lab.literature import europe_pmc_search

        response = europe_pmc_search('CRISPR AND OPEN_ACCESS:"Y"', result_type="lite", size=1)
        record = response["resultList"]["result"][0]

        assert record["isOpenAccess"] in {"Y", "N"}
        assert isinstance(record["pubYear"], str)
        assert isinstance(record["citedByCount"], int)

    @live_only
    def test_a_quoted_pmcid_still_matches_nothing(self) -> None:
        # The reason the identifier is sent unquoted. If this ever starts working, the shape
        # check can go; until then it is what makes the unquoted form safe.
        from virtual_lab.literature import europe_pmc_search

        quoted = europe_pmc_search('PMCID:"PMC3258128"', result_type="lite", size=1)
        plain = europe_pmc_search("PMCID:PMC3258128", result_type="lite", size=1)

        assert quoted["hitCount"] == 0
        assert plain["hitCount"] == 1

    @live_only
    def test_europe_pmc_still_answers_a_lookup_by_each_identifier(self) -> None:
        by_pmcid = get_article("PMC3258128")
        by_pmid = get_article("21937511")

        assert by_pmcid.pmcid == by_pmid.pmcid == "PMC3258128"
        assert by_pmcid.abstract

    @live_only
    def test_europe_pmc_still_returns_a_readable_full_text(self) -> None:
        text = get_article_text("PMC3258128")
        headings = [heading for heading, _ in text.sections]

        assert "Abstract" in headings
        assert any("RESULT" in heading.upper() for heading in headings)
        assert "REFERENCES" in text.skipped

    @live_only
    def test_asking_for_a_text_that_does_not_exist_still_costs_one_request(self) -> None:
        # The endpoint answers this with 500, which is why the record is checked first
        with pytest.raises(RecordNotFoundError):
            get_article_text("39366179")

    @live_only
    def test_arxiv_still_reads_loose_words_as_alternatives(self) -> None:
        # The entire reason arxiv_query exists. If arXiv ever changes this, the transformation
        # becomes unnecessary rather than wrong, but it should be noticed.
        loose = search_preprints("all:protein language model", limit=1)
        required = search_preprints("protein language model", limit=1)

        assert loose.total > 100 * required.total

    @live_only
    def test_arxiv_still_answers_a_search(self) -> None:
        results = search_preprints("protein language model", limit=3, category="q-bio.BM")

        assert len(results.preprints) == 3
        assert all(preprint.arxiv_id and preprint.title for preprint in results.preprints)
        assert all("q-bio.BM" in preprint.categories for preprint in results.preprints)

    @live_only
    def test_europe_pmc_still_does_not_index_arxiv(self) -> None:
        # Which is why there are two searches rather than one
        on_arxiv = search_preprints('"Attention Is All You Need"', limit=1)
        in_europe_pmc = search_articles('"Attention Is All You Need"', limit=5)

        assert on_arxiv.total >= 1
        assert not any(
            "arxiv" in (article.journal or "").lower() for article in in_europe_pmc.articles
        )
