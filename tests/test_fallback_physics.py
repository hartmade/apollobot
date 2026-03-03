"""Tests for physics fallback adapters (Materials Project, NIST, CERN Open Data)."""

import os
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from apollobot.mcp.fallback._physics import (
    _materials_project_search,
    _nist_search,
    _cern_opendata_search,
)


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

MP_JSON = {
    "data": [
        {
            "material_id": "mp-13",
            "formula_pretty": "Fe",
            "energy_above_hull": 0.0,
            "band_gap": 0.0,
            "density": 7.874,
        },
        {
            "material_id": "mp-19017",
            "formula_pretty": "Fe2O3",
            "energy_above_hull": 0.0,
            "band_gap": 2.1,
            "density": 5.24,
        },
    ]
}

NIST_TEXT = """\
  Fundamental Physical Constants --- Complete Listing

  Quantity                              Value                 Uncertainty          Unit
  ---------                             ------                -----------          ----
  speed of light in vacuum              299 792 458           (exact)              m s^-1
  Planck constant                       6.626 070 15 e-34     (exact)              J Hz^-1
"""

NIST_EMPTY = """No results found for the search query."""

CERN_JSON = {
    "hits": {
        "hits": [
            {
                "id": 1507,
                "metadata": {
                    "recid": 1507,
                    "title": "Higgs candidate events",
                    "experiment": "CMS",
                    "collision_information": {
                        "type": "pp",
                        "energy": "7TeV",
                    },
                },
            },
            {
                "id": 1508,
                "metadata": {
                    "recid": 1508,
                    "title": "Dimuon events",
                    "experiment": "CMS",
                    "collision_information": {},
                },
            },
        ]
    }
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
# Materials Project tests
# ---------------------------------------------------------------------------

class TestMaterialsProjectFallback:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MP_API_KEY": "test-key-123"})
    async def test_basic_search(self):
        http = _mock_http({
            "materials/summary": _response(json_data=MP_JSON),
        })
        result = await _materials_project_search(
            "https://api.materialsproject.org",
            {"query": "Fe2O3"},
            http,
        )
        materials = result["materials"]
        assert len(materials) == 2
        assert materials[0]["material_id"] == "mp-13"
        assert materials[0]["formula"] == "Fe"
        assert materials[0]["density"] == 7.874
        assert materials[0]["source"] == "materials-project"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MP_API_KEY": "test-key-123"})
    async def test_empty_results(self):
        http = _mock_http({
            "materials/summary": _response(json_data={"data": []}),
        })
        result = await _materials_project_search(
            "https://api.materialsproject.org",
            {"query": "zzzzz"},
            http,
        )
        assert result["materials"] == []

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Should raise ValueError when MP_API_KEY is not set."""
        http = _mock_http({})
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the key is not in env
            os.environ.pop("MP_API_KEY", None)
            with pytest.raises(ValueError, match="MP_API_KEY"):
                await _materials_project_search(
                    "https://api.materialsproject.org",
                    {"query": "Fe"},
                    http,
                )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MP_API_KEY": "test-key-123"})
    async def test_missing_fields(self):
        data = {"data": [{"material_id": "mp-1"}]}
        http = _mock_http({
            "materials/summary": _response(json_data=data),
        })
        result = await _materials_project_search(
            "https://api.materialsproject.org",
            {"query": "test"},
            http,
        )
        assert result["materials"][0]["formula"] == ""
        assert result["materials"][0]["band_gap"] is None


# ---------------------------------------------------------------------------
# NIST tests
# ---------------------------------------------------------------------------

class TestNistFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "Value": _response(text=NIST_TEXT),
        })
        result = await _nist_search(
            "https://physics.nist.gov/cgi-bin/cuu",
            {"query": "speed of light"},
            http,
        )
        constants = result["constants"]
        assert len(constants) > 0
        # Should find at least the speed of light
        names = [c["name"].lower() for c in constants]
        assert any("speed of light" in n for n in names)
        assert constants[0]["source"] == "nist"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "Value": _response(text=NIST_EMPTY),
        })
        result = await _nist_search(
            "https://physics.nist.gov/cgi-bin/cuu",
            {"query": "zzzzz"},
            http,
        )
        assert result["constants"] == []
        assert "note" in result

    @pytest.mark.asyncio
    async def test_non_200(self):
        http = _mock_http({
            "Value": _response(status=500),
        })
        result = await _nist_search(
            "https://physics.nist.gov/cgi-bin/cuu",
            {"query": "test"},
            http,
        )
        assert "note" in result


# ---------------------------------------------------------------------------
# CERN Open Data tests
# ---------------------------------------------------------------------------

class TestCernOpendataFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "records": _response(json_data=CERN_JSON),
        })
        result = await _cern_opendata_search(
            "https://opendata.cern.ch/api",
            {"query": "Higgs"},
            http,
        )
        datasets = result["datasets"]
        assert len(datasets) == 2
        assert datasets[0]["recid"] == 1507
        assert datasets[0]["title"] == "Higgs candidate events"
        assert datasets[0]["experiment"] == "CMS"
        assert datasets[0]["collision_type"] == "pp"
        assert datasets[0]["collision_energy"] == "7TeV"
        assert datasets[0]["source"] == "cern-opendata"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "records": _response(json_data={"hits": {"hits": []}}, status=200),
        })
        result = await _cern_opendata_search(
            "https://opendata.cern.ch/api",
            {"query": "zzzzz"},
            http,
        )
        assert result["datasets"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        http = _mock_http({
            "records": _response(json_data=CERN_JSON),
        })
        result = await _cern_opendata_search(
            "https://opendata.cern.ch/api",
            {"query": "test"},
            http,
        )
        # Second dataset has empty collision info
        assert result["datasets"][1]["collision_type"] == ""
        assert result["datasets"][1]["collision_energy"] == ""
