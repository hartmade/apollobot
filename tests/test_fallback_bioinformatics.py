"""Tests for bioinformatics fallback adapters (GEO, GenBank, UniProt, Ensembl, KEGG, PDB)."""

import pytest
import httpx
from unittest.mock import AsyncMock

from apollobot.mcp.fallback._bioinformatics import (
    _geo_search,
    _genbank_search,
    _uniprot_search,
    _ensembl_search,
    _kegg_search,
    _pdb_search,
)


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

GEO_SEARCH_JSON = {"esearchresult": {"idlist": ["200012345", "200067890"]}}

GEO_SUMMARY_JSON = {
    "result": {
        "200012345": {
            "accession": "GSE12345",
            "title": "RNA-seq of human tumors",
            "summary": "Transcriptome profiling of 50 tumor samples",
            "taxon": "Homo sapiens",
            "gpl": "GPL16791",
            "n_samples": "50",
        },
        "200067890": {
            "accession": "GSE67890",
            "title": "Mouse brain atlas",
            "summary": "Single-cell RNA-seq",
            "taxon": "Mus musculus",
            "gpl": "GPL24676",
            "n_samples": "1000",
        },
    }
}

GENBANK_SEARCH_JSON = {"esearchresult": {"idlist": ["111", "222"]}}

GENBANK_SUMMARY_JSON = {
    "result": {
        "111": {
            "caption": "NM_001301.4",
            "title": "Homo sapiens TP53 mRNA",
            "organism": "Homo sapiens",
            "slen": "2629",
            "moltype": "mRNA",
        },
        "222": {
            "caption": "NC_000001.11",
            "title": "Chromosome 1",
            "organism": "Homo sapiens",
            "slen": "248956422",
            "moltype": "DNA",
        },
    }
}

UNIPROT_JSON = {
    "results": [
        {
            "primaryAccession": "P04637",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Cellular tumor antigen p53"}
                }
            },
            "organism": {"scientificName": "Homo sapiens"},
            "genes": [{"geneName": {"value": "TP53"}}],
            "sequence": {"length": 393},
        },
        {
            "primaryAccession": "Q9Y6K9",
            "proteinDescription": {
                "submissionNames": [
                    {"fullName": {"value": "Unknown protein"}}
                ]
            },
            "organism": {"scientificName": "Homo sapiens"},
            "genes": [],
            "sequence": {"length": 100},
        },
    ]
}

ENSEMBL_JSON = [
    {
        "id": "ENSG00000141510",
        "description": "tumor protein p53",
        "type": "gene",
    },
    {
        "id": "ENSG00000141510",
        "description": "duplicate",
        "type": "gene",
    },
    {
        "id": "ENST00000269305",
        "description": "TP53 transcript",
        "type": "transcript",
    },
]

KEGG_TEXT = "hsa:7157\tTP53; tumor protein p53\nhsa:7158\tTP53I3; tumor protein p53 inducible protein 3\n"

PDB_SEARCH_JSON = {
    "result_set": [
        {"identifier": "1TUP"},
        {"identifier": "2XWR"},
    ]
}

PDB_ENTRY_1TUP = {
    "struct": {"title": "Crystal structure of p53 bound to DNA"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "reflns": [{"d_resolution_high": 2.2}],
    "rcsb_entry_info": {},
}

PDB_ENTRY_2XWR = {
    "struct": {"title": "Structure of p53 tetramer"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "reflns": [],
    "rcsb_entry_info": {"resolution_combined": [1.8]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_http(responses: dict[str, httpx.Response]):
    """Create a mock httpx client whose .get()/.post() dispatch by URL substring."""
    async def _get(url, params=None, **kwargs):
        for key, resp in responses.items():
            if key in str(url):
                return resp
        raise httpx.ConnectError(f"No mock for {url}")

    async def _post(url, **kwargs):
        for key, resp in responses.items():
            if key in str(url):
                return resp
        raise httpx.ConnectError(f"No mock for {url}")

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=_get)
    client.post = AsyncMock(side_effect=_post)
    return client


def _response(text: str = "", json_data: dict | list | None = None, status: int = 200):
    import json as _json
    resp = httpx.Response(status_code=status, request=httpx.Request("GET", "http://test"))
    if json_data is not None:
        resp._content = _json.dumps(json_data).encode()
    else:
        resp._content = text.encode()
    return resp


# ---------------------------------------------------------------------------
# GEO tests
# ---------------------------------------------------------------------------

class TestGeoFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "esearch": _response(json_data=GEO_SEARCH_JSON),
            "esummary": _response(json_data=GEO_SUMMARY_JSON),
        })
        result = await _geo_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "RNA-seq tumor"},
            http,
        )
        datasets = result["datasets"]
        assert len(datasets) == 2
        assert datasets[0]["accession"] == "GSE12345"
        assert datasets[0]["title"] == "RNA-seq of human tumors"
        assert datasets[0]["organism"] == "Homo sapiens"
        assert datasets[0]["sample_count"] == 50
        assert datasets[0]["source"] == "geo"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "esearch": _response(json_data={"esearchresult": {"idlist": []}}),
        })
        result = await _geo_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "zzzzz"},
            http,
        )
        assert result["datasets"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        summary = {"result": {"111": {"accession": "GSE111", "title": "Minimal"}}}
        http = _mock_http({
            "esearch": _response(json_data={"esearchresult": {"idlist": ["111"]}}),
            "esummary": _response(json_data=summary),
        })
        result = await _geo_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "test"},
            http,
        )
        assert result["datasets"][0]["organism"] == ""
        assert result["datasets"][0]["sample_count"] == 0


# ---------------------------------------------------------------------------
# GenBank tests
# ---------------------------------------------------------------------------

class TestGenbankFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "esearch": _response(json_data=GENBANK_SEARCH_JSON),
            "esummary": _response(json_data=GENBANK_SUMMARY_JSON),
        })
        result = await _genbank_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "TP53"},
            http,
        )
        seqs = result["sequences"]
        assert len(seqs) == 2
        assert seqs[0]["accession"] == "NM_001301.4"
        assert seqs[0]["organism"] == "Homo sapiens"
        assert seqs[0]["length"] == 2629
        assert seqs[0]["source"] == "genbank"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "esearch": _response(json_data={"esearchresult": {"idlist": []}}),
        })
        result = await _genbank_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "zzzzz"},
            http,
        )
        assert result["sequences"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        summary = {"result": {"111": {"caption": "ABC123"}}}
        http = _mock_http({
            "esearch": _response(json_data={"esearchresult": {"idlist": ["111"]}}),
            "esummary": _response(json_data=summary),
        })
        result = await _genbank_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            {"query": "test"},
            http,
        )
        assert result["sequences"][0]["accession"] == "ABC123"
        assert result["sequences"][0]["length"] == 0


# ---------------------------------------------------------------------------
# UniProt tests
# ---------------------------------------------------------------------------

class TestUniprotFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "uniprotkb/search": _response(json_data=UNIPROT_JSON),
        })
        result = await _uniprot_search("https://rest.uniprot.org", {"query": "p53"}, http)
        proteins = result["proteins"]
        assert len(proteins) == 2
        assert proteins[0]["accession"] == "P04637"
        assert proteins[0]["protein_name"] == "Cellular tumor antigen p53"
        assert proteins[0]["gene_names"] == ["TP53"]
        assert proteins[0]["length"] == 393
        assert proteins[0]["source"] == "uniprot"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "uniprotkb/search": _response(json_data={"results": []}),
        })
        result = await _uniprot_search("https://rest.uniprot.org", {"query": "zzzzz"}, http)
        assert result["proteins"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Protein with submissionNames instead of recommendedName."""
        http = _mock_http({
            "uniprotkb/search": _response(json_data=UNIPROT_JSON),
        })
        result = await _uniprot_search("https://rest.uniprot.org", {"query": "test"}, http)
        assert result["proteins"][1]["protein_name"] == "Unknown protein"
        assert result["proteins"][1]["gene_names"] == []


# ---------------------------------------------------------------------------
# Ensembl tests
# ---------------------------------------------------------------------------

class TestEnsemblFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "xrefs/symbol": _response(json_data=ENSEMBL_JSON),
        })
        result = await _ensembl_search(
            "https://rest.ensembl.org", {"query": "TP53"}, http
        )
        genes = result["genes"]
        # Should deduplicate by ID
        assert len(genes) == 2
        assert genes[0]["id"] == "ENSG00000141510"
        assert genes[0]["species"] == "homo_sapiens"
        assert genes[0]["source"] == "ensembl"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "xrefs/symbol": _response(json_data=[], status=400),
        })
        result = await _ensembl_search(
            "https://rest.ensembl.org", {"query": "ZZZZ"}, http
        )
        assert result["genes"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        data = [{"id": "ENS001"}]
        http = _mock_http({
            "xrefs/symbol": _response(json_data=data),
        })
        result = await _ensembl_search(
            "https://rest.ensembl.org", {"query": "X"}, http
        )
        assert result["genes"][0]["description"] == ""
        assert result["genes"][0]["biotype"] == ""


# ---------------------------------------------------------------------------
# KEGG tests
# ---------------------------------------------------------------------------

class TestKeggFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "find/pathway": _response(text=KEGG_TEXT),
        })
        result = await _kegg_search("https://rest.kegg.jp", {"query": "TP53"}, http)
        entries = result["entries"]
        assert len(entries) == 2
        assert entries[0]["id"] == "hsa:7157"
        assert "tumor protein p53" in entries[0]["description"]
        assert entries[0]["source"] == "kegg"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "find/pathway": _response(text="", status=404),
        })
        result = await _kegg_search("https://rest.kegg.jp", {"query": "zzzzz"}, http)
        assert result["entries"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        http = _mock_http({
            "find/pathway": _response(text="hsa:9999\n"),
        })
        result = await _kegg_search("https://rest.kegg.jp", {"query": "test"}, http)
        assert result["entries"][0]["id"] == "hsa:9999"
        assert result["entries"][0]["description"] == ""


# ---------------------------------------------------------------------------
# PDB tests
# ---------------------------------------------------------------------------

class TestPdbFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "rcsbsearch": _response(json_data=PDB_SEARCH_JSON),
            "core/entry/1TUP": _response(json_data=PDB_ENTRY_1TUP),
            "core/entry/2XWR": _response(json_data=PDB_ENTRY_2XWR),
        })
        result = await _pdb_search(
            "https://data.rcsb.org/rest/v1", {"query": "p53"}, http
        )
        structs = result["structures"]
        assert len(structs) == 2
        assert structs[0]["pdb_id"] == "1TUP"
        assert structs[0]["method"] == "X-RAY DIFFRACTION"
        assert structs[0]["resolution"] == 2.2
        assert structs[0]["source"] == "pdb"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "rcsbsearch": _response(json_data={"result_set": []}, status=200),
        })
        result = await _pdb_search(
            "https://data.rcsb.org/rest/v1", {"query": "zzzzz"}, http
        )
        assert result["structures"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Entry with resolution from rcsb_entry_info instead of reflns."""
        http = _mock_http({
            "rcsbsearch": _response(json_data={"result_set": [{"identifier": "2XWR"}]}),
            "core/entry/2XWR": _response(json_data=PDB_ENTRY_2XWR),
        })
        result = await _pdb_search(
            "https://data.rcsb.org/rest/v1", {"query": "test"}, http
        )
        assert result["structures"][0]["resolution"] == 1.8
