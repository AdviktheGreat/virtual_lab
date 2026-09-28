"""Lookups in the literature: published articles in Europe PMC, preprints on arXiv.

Two services rather than one because neither covers the field. Europe PMC indexes PubMed, PubMed
Central, and the biology preprint servers, but not arXiv, where most of the machine learning work
a computational biologist needs is published and stays. Searching one and calling it the
literature is how a plan comes back citing nothing written after the last review article.

The failure to guard against here is different from a database lookup's. A compound either exists
or it does not, but a search always returns something, and a search that quietly answered a
different question than the one asked still returns ten plausible titles. So the query a caller
wrote and the query the service ran are kept the same on purpose: arXiv reads unquoted words as
alternatives, which turns three terms into 1.2 million results, and Europe PMC reads them as
requirements. Both are documented behaviour and they are opposites.
"""

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from typing import Any

from virtual_lab.constants import (
    MAX_ABSTRACT_CHARACTERS,
    MAX_ARTICLE_CHARACTERS,
    MAX_AUTHORS_REPORTED,
    MAX_CATEGORIES_REPORTED,
    MAX_SEARCH_RESULTS,
    MAX_SECTION_CHARACTERS,
    SKIPPED_SECTION_TITLES,
)
from virtual_lab.records import (
    RecordNotFoundError,
    bounded,
    truncate_text,
)
from virtual_lab.web import WebRequestError, build_url, request_json, request_text

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ARXIV_BASE = "https://export.arxiv.org/api/query"

# Europe PMC answers an identifier it does not hold with 404 from the search endpoint. Its full
# text endpoint answers one with 500, which is handled where that request is made rather than
# here, since a 500 from anything else is what it says it is.
NOT_FOUND_STATUSES = frozenset({404})

# How a caller may ask for results to be ordered, and what Europe PMC calls it. Relevance is the
# service's default and is left empty rather than named, because naming it is not supported.
ARTICLE_SORTS = {
    "relevance": "",
    "cited": "CITED desc",
    "recent": "P_PDATE_D desc",
}

# The same for arXiv, which does name its default.
PREPRINT_SORTS = {
    "relevance": "relevance",
    "recent": "submittedDate",
    "updated": "lastUpdatedDate",
}

# Fields arXiv understands at the front of a term. A query already using one, or using a boolean
# operator, is the caller's own and is sent unchanged.
ARXIV_FIELDS = ("all:", "ti:", "au:", "abs:", "co:", "jr:", "cat:", "rn:", "id:")
ARXIV_OPERATORS = ("AND", "OR", "ANDNOT")

ATOM = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def yes(value: Any) -> bool:
    """Reads a flag that Europe PMC sends as the letter Y or the letter N.

    Not a detail. "N" is a non-empty string, so every one of these fields is true when tested
    directly, and an article marked as not open access reads as open access to any code that
    forgets. There are eleven such fields on a single search result.

    :param value: The value as the service sent it.
    :return: Whether it says yes.
    """
    return str(value).strip().upper() == "Y"


def collapse(text: str) -> str:
    """Puts a value that arrived wrapped across lines back on one line.

    :param text: The text as the service sent it.
    :return: The text with runs of whitespace reduced to single spaces.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def parse_xml(body: str) -> ElementTree.Element:
    """Parses a response as XML, refusing a document that declares entities.

    ElementTree resolves internal entities, and a document declaring a few nested ones expands to
    whatever size its author chose while the parser holds the result in memory. It is not a
    theoretical concern: the parser here does expand them, which was checked rather than assumed.
    Neither service has any reason to send a DOCTYPE, so the declaration is refused outright
    instead of the expansion being bounded.

    :param body: The response body.
    :raises WebRequestError: If the body declares a document type, or is not XML.
    :return: The root element.
    """
    if re.search(r"<!DOCTYPE", body[:4096], re.IGNORECASE):
        raise WebRequestError("Refusing to parse a response that declares a document type")

    try:
        return ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise WebRequestError(f"The response was not usable XML: {error}") from error


@dataclass(frozen=True)
class Article:
    """An article as Europe PMC holds it.

    :param article_id: The identifier within its source, which is what the URL is built from.
    :param source: Which collection it came from: MED for PubMed, PPR for a preprint.
    :param pmid: The PubMed identifier, if it has one.
    :param pmcid: The PubMed Central identifier, which is what full text is fetched by.
    :param doi: The DOI.
    :param title: The title.
    :param authors: The authors, in the order given.
    :param journal: The journal, or the preprint server for a preprint.
    :param year: The year of publication, which for a preprint may be the current one.
    :param cited_by: How many citations Europe PMC has counted.
    :param open_access: Whether the full text may be read without a subscription.
    :param in_europe_pmc: Whether Europe PMC holds the full text itself.
    :param abstract: The abstract, which a lite search does not return.
    """

    article_id: str
    source: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    title: str = ""
    authors: tuple[str, ...] = ()
    journal: str = ""
    year: int | None = None
    cited_by: int = 0
    open_access: bool = False
    in_europe_pmc: bool = False
    abstract: str = ""

    @property
    def is_preprint(self) -> bool:
        """Whether this is a preprint rather than a published article."""
        return self.source.upper() == "PPR"

    @property
    def has_full_text(self) -> bool:
        """Whether the full text can be fetched.

        Both conditions are needed. An article can be open access somewhere else on the web while
        Europe PMC holds only its abstract, and asking for the text of one of those is answered
        with a 500 rather than with anything that says so.
        """
        return bool(self.pmcid) and self.open_access and self.in_europe_pmc

    @property
    def url(self) -> str:
        """Where a person can read the record."""
        return f"https://europepmc.org/article/{self.source or 'MED'}/{self.article_id}"

    def credit(self) -> str:
        """Renders the authors, the journal, and the year as one citation-like phrase."""
        shown = list(self.authors[:MAX_AUTHORS_REPORTED])

        if len(self.authors) > len(shown):
            shown.append(f"and {len(self.authors) - len(shown)} others")

        parts = [", ".join(shown), self.journal, str(self.year) if self.year else ""]

        return ". ".join(part for part in parts if part)

    def summary(self) -> str:
        """Renders the article as the two lines a search result is worth."""
        marks = []

        if self.is_preprint:
            marks.append("preprint, not peer reviewed")

        if self.cited_by:
            marks.append(f"cited {self.cited_by:,}")

        if self.has_full_text:
            marks.append(f"full text: {self.pmcid}")

        noted = f" [{'; '.join(marks)}]" if marks else ""

        return f"  {self.title or 'untitled'}\n    {self.credit()}{noted}"

    def report(self) -> str:
        """Renders the article for a model."""
        lines = [self.title or "untitled", self.credit()]

        if self.is_preprint:
            lines.append("This is a preprint. It has not been peer reviewed.")

        identifiers = [
            f"{label}: {value}"
            for label, value in (("PMID", self.pmid), ("PMCID", self.pmcid), ("DOI", self.doi))
            if value
        ]

        if identifiers:
            lines.append(", ".join(identifiers))

        lines.append(f"Cited by {self.cited_by:,}.")

        if self.abstract:
            lines.append(f"\n{truncate_text(self.abstract, MAX_ABSTRACT_CHARACTERS)}")
        else:
            lines.append("\nNo abstract is recorded for this article.")

        if self.has_full_text:
            lines.append(f"\nFull text can be fetched with {self.pmcid}.")
        else:
            lines.append("\nEurope PMC does not hold the full text of this article.")

        lines.append(self.url)

        return "\n".join(lines)


@dataclass(frozen=True)
class ArticleResults:
    """What a Europe PMC search found.

    :param query: The query as it was sent.
    :param total: How many articles the service said match.
    :param articles: The articles kept, already limited.
    :param sort: How the results were ordered.
    """

    query: str
    total: int
    articles: tuple[Article, ...]
    sort: str = "relevance"

    def report(self) -> str:
        """Renders the results for a model."""
        if not self.articles:
            return (
                f'No articles in Europe PMC match "{self.query}". Europe PMC reads unquoted '
                f"words as terms that must all appear, so a long query finds less than a short "
                f"one. It does not index arXiv; use arxiv_search for that."
            )

        header = (
            f'{self.total:,} articles match "{self.query}", showing {len(self.articles)} '
            f"by {self.sort}:"
        )

        return "\n\n".join([header, *(article.summary() for article in self.articles)])


@dataclass(frozen=True)
class ArticleText:
    """The readable part of an article's full text.

    :param article: The article it belongs to.
    :param sections: Each section kept, as its heading and its text.
    :param skipped: The headings that were left out, so the omission is visible.
    """

    article: Article
    sections: tuple[tuple[str, str], ...] = ()
    skipped: tuple[str, ...] = ()

    def report(self) -> str:
        """Renders the text for a model, cut to what a meeting can be given."""
        lines = [self.article.title or "untitled", self.article.credit(), ""]
        spent = 0

        for heading, text in self.sections:
            if spent >= MAX_ARTICLE_CHARACTERS:
                lines.append("\n(The rest of the article is not shown.)")
                break

            room = min(MAX_SECTION_CHARACTERS, MAX_ARTICLE_CHARACTERS - spent)
            shown = truncate_text(text, room)
            spent += len(shown)
            lines.append(f"\n## {heading}\n{shown}")

        if self.skipped:
            lines.append(f"\n(Sections not shown: {', '.join(self.skipped)}.)")

        lines.append(f"\n{self.article.url}")

        return "\n".join(lines)


@dataclass(frozen=True)
class Preprint:
    """A preprint as arXiv holds it.

    :param arxiv_id: The identifier with its version, such as 2410.08355v3.
    :param title: The title.
    :param authors: The authors, in the order given.
    :param abstract: The abstract, which arXiv calls the summary.
    :param published: The date first submitted.
    :param updated: The date of the version returned.
    :param categories: The subject categories, the first being the primary one.
    :param comment: The author's note, which is where acceptance at a venue is usually recorded.
    :param journal_reference: Where it was published, if the authors said.
    :param doi: The DOI of the published version, if the authors gave one.
    """

    arxiv_id: str
    title: str = ""
    authors: tuple[str, ...] = ()
    abstract: str = ""
    published: str = ""
    updated: str = ""
    categories: tuple[str, ...] = ()
    comment: str = ""
    journal_reference: str = ""
    doi: str = ""

    @property
    def url(self) -> str:
        """Where a person can read the preprint."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    def credit(self) -> str:
        """Renders the authors and the date as one phrase."""
        shown = list(self.authors[:MAX_AUTHORS_REPORTED])

        if len(self.authors) > len(shown):
            shown.append(f"and {len(self.authors) - len(shown)} others")

        return ". ".join(part for part in (", ".join(shown), self.published[:10]) if part)

    def summary(self) -> str:
        """Renders the preprint as the two lines a search result is worth."""
        marks = list(self.categories[:MAX_CATEGORIES_REPORTED])

        # The note is where "accepted at NeurIPS 2024" lives, which is the only thing in the
        # record that says whether anyone but the authors has read it
        if self.journal_reference:
            marks.append(f"published: {collapse(self.journal_reference)}")
        elif self.comment:
            marks.append(collapse(self.comment))

        noted = f" [{'; '.join(marks)}]" if marks else ""

        return f"  {self.title or 'untitled'}\n    {self.credit()}{noted}\n    {self.url}"

    def report(self) -> str:
        """Renders the preprint for a model."""
        lines = [self.title or "untitled", self.credit()]

        if self.updated and self.updated[:10] != self.published[:10]:
            lines.append(f"Revised {self.updated[:10]}.")

        if self.categories:
            lines.append(f"Categories: {', '.join(self.categories[:MAX_CATEGORIES_REPORTED])}")

        if self.journal_reference:
            lines.append(f"Published as: {collapse(self.journal_reference)}")
        elif self.comment:
            lines.append(f"Author's note: {collapse(self.comment)}")

        if self.doi:
            lines.append(f"DOI: {self.doi}")

        lines.append(
            "This is a preprint. Unless a journal reference above says otherwise, it has not "
            "been peer reviewed."
        )

        if self.abstract:
            lines.append(f"\n{truncate_text(self.abstract, MAX_ABSTRACT_CHARACTERS)}")

        lines.append(f"\n{self.url}")

        return "\n".join(lines)


@dataclass(frozen=True)
class PreprintResults:
    """What an arXiv search found.

    :param query: The query as it was sent, which may not be the query as it was written.
    :param asked: The query as the caller wrote it.
    :param total: How many preprints arXiv said match.
    :param preprints: The preprints kept, already limited.
    """

    query: str
    asked: str
    total: int
    preprints: tuple[Preprint, ...] = ()

    def report(self) -> str:
        """Renders the results for a model."""
        if not self.preprints:
            return f'No preprints on arXiv match "{self.asked}".'

        header = f'{self.total:,} preprints match "{self.asked}", showing {len(self.preprints)}:'

        # Said out loud, because a caller who wrote three words and is shown a count taken from
        # a different query has no way to tell that from the results
        if self.query != self.asked:
            header += (
                f"\n(Searched as {self.query}, since arXiv reads loose words as alternatives "
                f"rather than as requirements.)"
            )

        return "\n\n".join([header, *(preprint.summary() for preprint in self.preprints)])


def article_from(record: dict[str, Any]) -> Article:
    """Builds an article record from one Europe PMC search result.

    :param record: One entry of the result list.
    :return: The article.
    """
    authors = tuple(
        collapse(str(author.get("fullName") or ""))
        for author in ((record.get("authorList") or {}).get("author") or [])
        if author.get("fullName")
    )

    # A lite search does not return the author list, only the string it was rendered from, and a
    # search result with no named authors would otherwise be reported as having none. The string
    # ends in a full stop, which becomes part of the last author's name unless it is taken off.
    if not authors and record.get("authorString"):
        authors = tuple(
            stripped
            for part in str(record["authorString"]).split(",")
            if (stripped := part.strip().rstrip("."))
        )

    # A preprint has no journal. Its server is recorded under the book details instead, and
    # leaving it blank is what makes a bioRxiv preprint read like a paper with no venue.
    journal = collapse(
        str(record.get("journalTitle") or "")
        or str(((record.get("journalInfo") or {}).get("journal") or {}).get("title") or "")
        or str((record.get("bookOrReportDetails") or {}).get("publisher") or "")
    )

    year = record.get("pubYear")

    return Article(
        article_id=str(record.get("id") or ""),
        source=str(record.get("source") or ""),
        pmid=str(record.get("pmid") or ""),
        pmcid=str(record.get("pmcid") or ""),
        doi=str(record.get("doi") or ""),
        title=collapse(str(record.get("title") or "")),
        authors=authors,
        journal=journal,
        # Sent as a string, unlike citedByCount beside it, which is sent as a number
        year=int(year) if str(year or "").strip().isdigit() else None,
        cited_by=int(record.get("citedByCount") or 0),
        open_access=yes(record.get("isOpenAccess")),
        in_europe_pmc=yes(record.get("inEPMC")),
        abstract=collapse(str(record.get("abstractText") or "")),
    )


def europe_pmc_search(query: str, result_type: str, size: int, sort: str = "") -> dict[str, Any]:
    """Runs one Europe PMC search.

    :param query: The query, sent as a parameter rather than built into the path.
    :param result_type: "lite" for a listing, "core" for the abstract and the rest.
    :param size: How many results to ask for.
    :param sort: The service's name for an ordering, empty for its default.
    :raises WebRequestError: If Europe PMC could not be reached.
    :return: The parsed response.
    """
    params: dict[str, Any] = {
        "query": query,
        "format": "json",
        "resultType": result_type,
        "pageSize": size,
    }

    if sort:
        params["sort"] = sort

    response = request_json(f"{EUROPE_PMC_BASE}/search", params=params)

    return response if isinstance(response, dict) else {}


def search_articles(
    query: str,
    limit: int = 10,
    sort: str = "relevance",
    open_access_only: bool = False,
    include_preprints: bool = True,
) -> ArticleResults:
    """Searches Europe PMC for published articles and biology preprints.

    :param query: What to search for. Unquoted words must all appear; quote a phrase to require
        the words together.
    :param limit: How many results to return.
    :param sort: "relevance", "cited", or "recent".
    :param open_access_only: Only return articles whose full text can then be fetched.
    :param include_preprints: Whether to include preprint servers such as bioRxiv.
    :raises ValueError: If the query is empty, the limit is below one, or the sort is unknown.
    :raises WebRequestError: If Europe PMC could not be reached.
    :return: The results.
    """
    if not query.strip():
        raise ValueError("Give something to search for")

    if sort not in ARTICLE_SORTS:
        raise ValueError(f'Unknown sort "{sort}". Use one of: {", ".join(ARTICLE_SORTS)}.')

    size = bounded(limit, MAX_SEARCH_RESULTS, "limit")

    asked = query.strip()
    filters = []

    # Both filters are applied by the service rather than by dropping results here, since
    # filtering afterwards would return fewer than the caller asked for and not say why
    if open_access_only:
        filters.append('(OPEN_ACCESS:"Y" AND IN_EPMC:"Y")')

    if not include_preprints:
        filters.append('NOT SRC:"PPR"')

    # The caller's query is bracketed before anything is joined to it, because AND binds tighter
    # than OR. Without the brackets "a OR b" with a filter appended is read as "a OR (b AND
    # filter)", which returns everything matching a whether it passes the filter or not.
    sent = " AND ".join([f"({asked})", *filters]) if filters else asked

    response = europe_pmc_search(
        query=sent,
        result_type="lite",
        size=size,
        sort=ARTICLE_SORTS[sort],
    )

    records = ((response.get("resultList") or {}).get("result") or [])[:size]

    return ArticleResults(
        query=asked,
        total=int(response.get("hitCount") or 0),
        articles=tuple(article_from(record) for record in records),
        sort=sort,
    )


def get_article(identifier: str) -> Article:
    """Looks one article up in Europe PMC by PMID, PMCID, or DOI.

    :param identifier: A PubMed identifier, a PMCID such as PMC3258128, or a DOI.
    :raises ValueError: If no identifier is given.
    :raises RecordNotFoundError: If Europe PMC holds no such article.
    :raises WebRequestError: If Europe PMC could not be reached.
    :return: The article, with its abstract.
    """
    wanted = identifier.strip()

    if not wanted:
        raise ValueError("Give a PMID, a PMCID, or a DOI to look up")

    # Quoting is not a matter of taste here. PMCID:"PMC3258128" and EXT_ID:"39366179" both match
    # nothing at all, while the same queries unquoted match the article; DOI needs the quotes,
    # because a DOI carries slashes and parentheses that are read as query syntax without them.
    # So the two that must go unquoted are checked against their shape first, which is what makes
    # sending them unquoted safe: neither can then carry a character that means anything here.
    if re.fullmatch(r"(?i)PMC\d+", wanted):
        query = f"PMCID:{wanted.upper()}"
    elif wanted.isdigit():
        query = f"EXT_ID:{wanted} AND SRC:MED"
    elif '"' in wanted:
        raise ValueError(f'"{wanted}" is not a PMID, a PMCID, or a DOI')
    else:
        query = f'DOI:"{wanted}"'

    response = europe_pmc_search(query=query, result_type="core", size=1)
    records = (response.get("resultList") or {}).get("result") or []

    if not records:
        raise RecordNotFoundError(f'Europe PMC has no article for "{wanted}".')

    return article_from(records[0])


def section_text(element: ElementTree.Element) -> str:
    """Reads the readable text of one JATS section, without the heading that names it.

    The heading is the first thing itertext() reaches, so taking the text wholesale produces
    "AbstractmicroRNAs are a versatile class..." under a heading that already said Abstract. Only
    the section's own title is dropped; a subsection keeps its own, since those are the structure
    of the argument rather than a repeat.

    :param element: The section element.
    :return: Its text, with runs of whitespace collapsed.
    """
    parts: list[str] = [element.text or ""]
    dropped = False

    for child in element:
        if child.tag == "title" and not dropped:
            dropped = True
        elif child.tag == "sec":
            # Recursed into rather than read wholesale, so that a subsection's heading is marked
            # off from its first sentence instead of reading as the start of it
            heading = collapse(child.findtext("title", ""))
            nested = section_text(child)
            parts.append(f" {heading}. {nested}" if heading else f" {nested}")
        else:
            parts.append("".join(child.itertext()))

        parts.append(child.tail or "")

    return collapse("".join(parts))


def is_skipped(heading: str) -> bool:
    """Whether a section heading names something not worth returning.

    :param heading: The section's title.
    :return: Whether to leave it out.
    """
    lowered = heading.strip().lower()

    return any(lowered.startswith(skipped) for skipped in SKIPPED_SECTION_TITLES)


def get_article_text(pmcid: str) -> ArticleText:
    """Fetches the full text of an open access article held by Europe PMC.

    The article is looked up before its text is requested, which is not an extra round trip for
    its own sake. Europe PMC answers a request for the text of an article it does not hold with
    HTTP 500, the same status a genuine outage gives, so the retry logic spends three attempts
    and several seconds of backoff and then reports that the service could not be reached. The
    record says plainly whether the text exists, so it is asked first.

    :param pmcid: A PubMed Central identifier such as PMC3258128.
    :raises ValueError: If no identifier is given.
    :raises RecordNotFoundError: If there is no such article, or its full text is not open.
    :raises WebRequestError: If Europe PMC could not be reached.
    :return: The article's sections.
    """
    wanted = pmcid.strip().upper()

    if not wanted:
        raise ValueError("Give a PMCID such as PMC3258128")

    article = get_article(wanted)

    if not article.has_full_text:
        raise RecordNotFoundError(
            f"Europe PMC holds no open full text for {wanted}. Its abstract is available through "
            f"europepmc_lookup, and the article itself may be readable at {article.url}."
        )

    url = build_url(f"{EUROPE_PMC_BASE}/{{pmcid}}/fullTextXML", pmcid=article.pmcid)
    root = parse_xml(request_text(url))

    sections: list[tuple[str, str]] = []
    skipped: list[str] = []

    if (abstract := root.find(".//abstract")) is not None:
        if text := section_text(abstract):
            sections.append(("Abstract", text))

    for element in root.findall("./body/sec"):
        heading = collapse(element.findtext("title", "")) or "Untitled section"

        if is_skipped(heading):
            skipped.append(heading)
            continue

        if text := section_text(element):
            sections.append((heading, text))

    return ArticleText(article=article, sections=tuple(sections), skipped=tuple(skipped))


def arxiv_query(query: str) -> str:
    """Turns what a caller wrote into what arXiv should be asked.

    arXiv reads a term with no field prefix as a match on any field and, crucially, reads several
    of them as alternatives. "protein language model" returns 1,285,358 results, being everything
    mentioning any of the three words, while requiring all three returns 893 and the phrase
    returns 344. The first is not a useful answer to anything, and the count itself misleads: a
    caller told there are a million results concludes the field is enormous.

    A query that already uses a field prefix or a boolean operator is the caller's own and is
    passed through, so this cannot get in the way of someone who knows the syntax.

    :param query: The query as written.
    :return: The query to send.
    """
    asked = collapse(query)

    # Quoted phrases kept whole and in place: the words inside one are not separate terms, and
    # tokenising in order keeps the query recognisable to whoever wrote it
    terms = [match.group(0) for match in re.finditer(r'"[^"]*"|\S+', asked)]

    # Tested against the start of a term rather than anywhere in the query. As a substring,
    # "id:" matches "covid:" and would hand a plain query straight through unchanged.
    if any(term.lower().startswith(ARXIV_FIELDS) for term in terms):
        return asked

    if any(term.upper() in ARXIV_OPERATORS for term in terms):
        return asked

    if len(terms) <= 1:
        return f"all:{terms[0]}" if terms else asked

    return " AND ".join(f"all:{term}" for term in terms)


def preprint_from(entry: ElementTree.Element) -> Preprint:
    """Builds a preprint record from one arXiv Atom entry.

    :param entry: The entry element.
    :return: The preprint.
    """
    # The identifier arrives as a URL, and the version on the end matters: a result for v1 of a
    # paper that is now on v3 is a different document
    raw_id = collapse(entry.findtext("atom:id", "", ATOM))
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id

    categories = tuple(
        dict.fromkeys(
            term
            for element in entry.findall("atom:category", ATOM)
            if (term := element.attrib.get("term", "").strip())
        )
    )

    return Preprint(
        arxiv_id=arxiv_id,
        # Wrapped across lines in the feed, so a title used unchanged carries newlines into the
        # middle of a report
        title=collapse(entry.findtext("atom:title", "", ATOM)),
        authors=tuple(
            collapse(name)
            for author in entry.findall("atom:author", ATOM)
            if (name := author.findtext("atom:name", "", ATOM))
        ),
        abstract=collapse(entry.findtext("atom:summary", "", ATOM)),
        published=collapse(entry.findtext("atom:published", "", ATOM)),
        updated=collapse(entry.findtext("atom:updated", "", ATOM)),
        categories=categories,
        comment=collapse(entry.findtext("arxiv:comment", "", ATOM)),
        journal_reference=collapse(entry.findtext("arxiv:journal_ref", "", ATOM)),
        doi=collapse(entry.findtext("arxiv:doi", "", ATOM)),
    )


def search_preprints(
    query: str,
    limit: int = 10,
    category: str = "",
    sort: str = "relevance",
) -> PreprintResults:
    """Searches arXiv for preprints.

    Worth having beside the Europe PMC search rather than folded into it, because Europe PMC does
    not index arXiv at all. A question about a protein language model, a docking score predictor,
    or anything else where the method came from machine learning is answered mostly by papers
    that exist only here.

    :param query: What to search for.
    :param limit: How many results to return.
    :param category: Restrict to one arXiv category, such as q-bio.BM or cs.LG.
    :param sort: "relevance", "recent", or "updated".
    :raises ValueError: If the query is empty, the limit is below one, or the sort is unknown.
    :raises WebRequestError: If arXiv could not be reached, or sent something unparseable.
    :return: The results.
    """
    if not query.strip():
        raise ValueError("Give something to search for")

    if sort not in PREPRINT_SORTS:
        raise ValueError(f'Unknown sort "{sort}". Use one of: {", ".join(PREPRINT_SORTS)}.')

    size = bounded(limit, MAX_SEARCH_RESULTS, "limit")

    asked = collapse(query)
    sent = arxiv_query(asked)

    if category.strip():
        sent = f"({sent}) AND cat:{collapse(category)}"

    body = request_text(
        ARXIV_BASE,
        params={
            "search_query": sent,
            "max_results": size,
            "sortBy": PREPRINT_SORTS[sort],
            "sortOrder": "descending",
        },
    )
    root = parse_xml(body)

    total = collapse(root.findtext("opensearch:totalResults", "", ATOM))
    entries = root.findall("atom:entry", ATOM)[:size]
    preprints = tuple(preprint_from(entry) for entry in entries)

    return PreprintResults(
        query=sent,
        asked=asked,
        total=int(total) if total.isdigit() else len(preprints),
        preprints=preprints,
    )
