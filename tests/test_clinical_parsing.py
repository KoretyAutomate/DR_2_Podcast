"""Tests for JSON parsing and PubMed XML parsing in clinical.py.

File under test: dr2_podcast/research/clinical.py
  - ResearchAgent._parse_json_response()
  - PubMedClient._parse_articles_xml()
"""

import json

import pytest

from dr2_podcast.research.clinical import ResearchAgent, PubMedClient


# ---------------------------------------------------------------------------
# ResearchAgent._parse_json_response
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Create a bare ResearchAgent instance (no clients needed for parsing)."""
    a = ResearchAgent.__new__(ResearchAgent)
    return a


class TestParseJsonResponse:

    def test_clean_json(self, agent):
        result = agent._parse_json_response('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_in_code_block(self, agent):
        raw = '```json\n{"key": "value"}\n```'
        result = agent._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_leading_text_before_json(self, agent):
        raw = 'Here is the JSON:\n{"result": true}'
        result = agent._parse_json_response(raw)
        assert result == {"result": True}

    def test_truncated_json_suffix_repair(self, agent):
        raw = '{"key": "value", "items": [1, 2, 3]'
        result = agent._parse_json_response(raw)
        assert result["key"] == "value"
        assert result["items"] == [1, 2, 3]

    def test_non_json_text_raises(self, agent):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            agent._parse_json_response("This is just plain text with no braces.")

    def test_empty_string_raises(self, agent):
        with pytest.raises(ValueError, match="Empty response"):
            agent._parse_json_response("")

    def test_nested_json(self, agent):
        raw = '{"outer": {"inner": {"deep": true}}}'
        result = agent._parse_json_response(raw)
        assert result["outer"]["inner"]["deep"] is True

    def test_json_array(self, agent):
        raw = '[{"a": 1}, {"b": 2}]'
        result = agent._parse_json_response(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_template_echo_nulled_out(self, agent):
        raw = json.dumps({
            "study_design": "parallel RCT | crossover RCT | meta-analysis",
            "sample_size": "200",
        })
        result = agent._parse_json_response(raw)
        assert result["study_design"] is None
        assert result["sample_size"] == "200"

    def test_think_blocks_stripped(self, agent):
        raw = '<think>Some thinking here</think>\n{"answer": "clean"}'
        result = agent._parse_json_response(raw)
        assert result["answer"] == "clean"


# ---------------------------------------------------------------------------
# PubMedClient._parse_articles_xml
# ---------------------------------------------------------------------------

MINIMAL_ARTICLE_XML = """\
<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Test Study Title</ArticleTitle>
        <Abstract>
          <AbstractText>This is the abstract.</AbstractText>
        </Abstract>
        <Journal><Title>Test Journal</Title></Journal>
        <ELocationID EIdType="doi">10.1000/test</ELocationID>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Caffeine</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

TWO_ARTICLES_XML = """\
<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <ArticleTitle>Study One</ArticleTitle>
        <Abstract><AbstractText>Abstract one.</AbstractText></Abstract>
        <Journal><Title>Journal A</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">11111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <ArticleTitle>Study Two</ArticleTitle>
        <Abstract><AbstractText>Abstract two.</AbstractText></Abstract>
        <Journal><Title>Journal B</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">22222222</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

STRUCTURED_ABSTRACT_XML = """\
<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333333</PMID>
      <Article>
        <ArticleTitle>Structured Abstract Study</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText Label="METHODS">Methods text.</AbstractText>
          <AbstractText Label="RESULTS">Results text.</AbstractText>
        </Abstract>
        <Journal><Title>Structured J</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">33333333</ArticleId>
        <ArticleId IdType="doi">10.2000/structured</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

NO_ABSTRACT_XML = """\
<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>44444444</PMID>
      <Article>
        <ArticleTitle>No Abstract Study</ArticleTitle>
        <Journal><Title>Minimal J</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">44444444</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

EMPTY_ARTICLES_XML = """\
<?xml version="1.0" ?>
<PubmedArticleSet>
</PubmedArticleSet>
"""


@pytest.fixture
def pubmed_client():
    return PubMedClient()


class TestParseArticlesXml:

    def test_two_articles(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(TWO_ARTICLES_XML)
        assert len(articles) == 2
        pmids = {a["pmid"] for a in articles}
        assert "11111111" in pmids
        assert "22222222" in pmids

    def test_structured_abstract(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(STRUCTURED_ABSTRACT_XML)
        assert len(articles) == 1
        abstract = articles[0]["abstract"]
        assert "BACKGROUND:" in abstract
        assert "METHODS:" in abstract
        assert "RESULTS:" in abstract

    def test_doi_from_article_id(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(STRUCTURED_ABSTRACT_XML)
        assert articles[0]["doi"] == "10.2000/structured"

    def test_doi_from_elocation_id(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(MINIMAL_ARTICLE_XML)
        assert articles[0]["doi"] == "10.1000/test"

    def test_missing_abstract(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(NO_ABSTRACT_XML)
        assert len(articles) == 1
        assert articles[0]["abstract"] == ""

    def test_mesh_headings_extracted(self, pubmed_client):
        articles = pubmed_client._parse_articles_xml(MINIMAL_ARTICLE_XML)
        assert "Caffeine" in articles[0]["mesh_headings"]

    def test_malformed_xml_returns_empty(self, pubmed_client):
        result = pubmed_client._parse_articles_xml("<broken xml!!!>>>>>")
        assert result == []

    def test_empty_article_set(self, pubmed_client):
        result = pubmed_client._parse_articles_xml(EMPTY_ARTICLES_XML)
        assert result == []
