"""Tests for computational chemistry fallback adapters (PubChem, ChEMBL, AlphaFold, ZINC)."""

import pytest
import httpx
from unittest.mock import AsyncMock

from apollobot.mcp.fallback._comp_chem import (
    _pubchem_search,
    _chembl_search,
    _alphafold_search,
    _zinc_search,
)


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

PUBCHEM_JSON = {
    "PC_Compounds": [
        {
            "id": {"id": {"cid": 2244}},
            "props": [
                {
                    "urn": {"label": "IUPAC Name", "name": "Preferred"},
                    "value": {"sval": "2-acetoxybenzoic acid"},
                },
                {
                    "urn": {"label": "Molecular Formula"},
                    "value": {"sval": "C9H8O4"},
                },
                {
                    "urn": {"label": "Molecular Weight"},
                    "value": {"fval": 180.16},
                },
                {
                    "urn": {"label": "SMILES", "name": "Canonical"},
                    "value": {"sval": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                },
            ],
        }
    ]
}

CHEMBL_JSON = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL25",
            "pref_name": "ASPIRIN",
            "molecule_type": "Small molecule",
            "max_phase": 4,
        },
        {
            "molecule_chembl_id": "CHEMBL1234",
            "pref_name": None,
            "molecule_type": "Small molecule",
            "max_phase": "2",
        },
    ]
}

ALPHAFOLD_JSON = [
    {
        "entryId": "AF-P04637-F1",
        "uniprotAccession": "P04637",
        "gene": "TP53",
        "organismScientificName": "Homo sapiens",
        "globalMetricValue": 70.5,
    }
]

ZINC_JSON = {
    "substances": [
        {
            "zinc_id": "ZINC000000000001",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "mwt": 180.16,
            "logp": 1.2,
        },
        {
            "zinc_id": "ZINC000000000002",
            "smiles": "C1=CC=CC=C1",
            "mwt": 78.11,
            "logp": 1.56,
        },
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_http(responses: dict[str, httpx.Response]):
    async def _get(url, params=None, **kwargs):
        for key, resp in responses.items():
            if key in str(url):
                return resp
        raise httpx.ConnectError(f"No mock for {url}")

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=_get)
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
# PubChem tests
# ---------------------------------------------------------------------------

class TestPubchemFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "compound/name/aspirin": _response(json_data=PUBCHEM_JSON),
        })
        result = await _pubchem_search(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            {"query": "aspirin"},
            http,
        )
        compounds = result["compounds"]
        assert len(compounds) == 1
        assert compounds[0]["cid"] == 2244
        assert compounds[0]["name"] == "2-acetoxybenzoic acid"
        assert compounds[0]["formula"] == "C9H8O4"
        assert compounds[0]["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
        assert compounds[0]["source"] == "pubchem"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "compound/name/zzzzz": _response(status=404),
        })
        result = await _pubchem_search(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            {"query": "zzzzz"},
            http,
        )
        assert result["compounds"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        minimal = {"PC_Compounds": [{"id": {"id": {"cid": 999}}, "props": []}]}
        http = _mock_http({
            "compound/name/x": _response(json_data=minimal),
        })
        result = await _pubchem_search(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            {"query": "x"},
            http,
        )
        assert result["compounds"][0]["cid"] == 999
        assert result["compounds"][0]["formula"] == ""


# ---------------------------------------------------------------------------
# ChEMBL tests
# ---------------------------------------------------------------------------

class TestChemblFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "molecule/search": _response(json_data=CHEMBL_JSON),
        })
        result = await _chembl_search(
            "https://www.ebi.ac.uk/chembl/api/data",
            {"query": "aspirin"},
            http,
        )
        mols = result["molecules"]
        assert len(mols) == 2
        assert mols[0]["chembl_id"] == "CHEMBL25"
        assert mols[0]["pref_name"] == "ASPIRIN"
        assert mols[0]["max_phase"] == 4
        assert mols[0]["source"] == "chembl"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "molecule/search": _response(json_data={"molecules": []}, status=200),
        })
        result = await _chembl_search(
            "https://www.ebi.ac.uk/chembl/api/data",
            {"query": "zzzzz"},
            http,
        )
        assert result["molecules"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        http = _mock_http({
            "molecule/search": _response(json_data=CHEMBL_JSON),
        })
        result = await _chembl_search(
            "https://www.ebi.ac.uk/chembl/api/data",
            {"query": "test"},
            http,
        )
        # max_phase as string should be safely converted
        assert result["molecules"][1]["max_phase"] == 2
        assert result["molecules"][1]["pref_name"] is None


# ---------------------------------------------------------------------------
# AlphaFold tests
# ---------------------------------------------------------------------------

class TestAlphafoldFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "prediction/P04637": _response(json_data=ALPHAFOLD_JSON),
        })
        result = await _alphafold_search(
            "https://alphafold.ebi.ac.uk/api",
            {"query": "P04637"},
            http,
        )
        preds = result["predictions"]
        assert len(preds) == 1
        assert preds[0]["entry_id"] == "AF-P04637-F1"
        assert preds[0]["gene"] == "TP53"
        assert preds[0]["confidence"] == 70.5
        assert preds[0]["source"] == "alphafold-db"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "prediction/ZZZZ": _response(status=404),
        })
        result = await _alphafold_search(
            "https://alphafold.ebi.ac.uk/api",
            {"query": "ZZZZ"},
            http,
        )
        assert result["predictions"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        http = _mock_http({
            "prediction/": _response(json_data={"predictions": []}),
        })
        result = await _alphafold_search(
            "https://alphafold.ebi.ac.uk/api",
            {},
            http,
        )
        assert result["predictions"] == []


# ---------------------------------------------------------------------------
# ZINC tests
# ---------------------------------------------------------------------------

class TestZincFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "substances/search": _response(json_data=ZINC_JSON),
        })
        result = await _zinc_search(
            "https://zinc.docking.org/api",
            {"query": "aspirin"},
            http,
        )
        subs = result["substances"]
        assert len(subs) == 2
        assert subs[0]["zinc_id"] == "ZINC000000000001"
        assert subs[0]["mwt"] == 180.16
        assert subs[0]["source"] == "zinc"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "substances/search": _response(json_data={"substances": []}, status=200),
        })
        result = await _zinc_search(
            "https://zinc.docking.org/api",
            {"query": "zzzzz"},
            http,
        )
        assert result["substances"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        data = {"substances": [{"id": "Z1"}]}
        http = _mock_http({
            "substances/search": _response(json_data=data),
        })
        result = await _zinc_search(
            "https://zinc.docking.org/api",
            {"query": "test"},
            http,
        )
        assert result["substances"][0]["zinc_id"] == "Z1"
        assert result["substances"][0]["smiles"] == ""
