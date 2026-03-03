"""Tests for economics fallback adapters (FRED, World Bank, BLS, SEC EDGAR)."""

import os
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from apollobot.mcp.fallback._economics import (
    _fred_search,
    _world_bank_search,
    _bls_search,
    _sec_edgar_search,
)


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

FRED_SEARCH_JSON = {
    "seriess": [
        {
            "id": "GDP",
            "title": "Gross Domestic Product",
            "frequency": "Quarterly",
            "units": "Billions of Dollars",
        },
        {
            "id": "UNRATE",
            "title": "Unemployment Rate",
            "frequency": "Monthly",
            "units": "Percent",
        },
    ]
}

FRED_OBS_JSON = {
    "observations": [
        {"date": "2025-01-01", "value": "28000.5"},
        {"date": "2024-10-01", "value": "27500.2"},
    ]
}

WB_INDICATOR_SEARCH_JSON = [
    {"page": 1, "pages": 1, "total": 2},
    [
        {"id": "NY.GDP.MKTP.CD", "name": "GDP (current US$)"},
        {"id": "SP.POP.TOTL", "name": "Population, total"},
    ],
]

WB_DATA_JSON = [
    {"page": 1, "pages": 1, "total": 2},
    [
        {
            "country": {"value": "United States"},
            "indicator": {"value": "GDP (current US$)"},
            "date": "2023",
            "value": 25462700000000,
        },
        {
            "country": {"value": "United States"},
            "indicator": {"value": "GDP (current US$)"},
            "date": "2022",
            "value": 25035200000000,
        },
    ],
]

BLS_JSON = {
    "Results": {
        "series": [
            {
                "seriesID": "CES0000000001",
                "data": [
                    {"year": "2025", "period": "M01", "value": "157500"},
                    {"year": "2024", "period": "M12", "value": "157200"},
                ],
            }
        ]
    }
}

EDGAR_JSON = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "file_num": "001-12345",
                    "display_names": ["Apple Inc."],
                    "form_type": "10-K",
                    "file_date": "2025-01-15",
                }
            },
            {
                "_source": {
                    "accession_no": "002-67890",
                    "entity_name": "Google LLC",
                    "file_type": "10-Q",
                    "period_of_report": "2024-12-31",
                }
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
# FRED tests
# ---------------------------------------------------------------------------

class TestFredFallback:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"FRED_API_KEY": "test-fred-key"})
    async def test_basic_search(self):
        http = _mock_http({
            "series/search": _response(json_data=FRED_SEARCH_JSON),
        })
        result = await _fred_search(
            "https://api.stlouisfed.org/fred",
            {"query": "GDP"},
            http,
        )
        series = result["series"]
        assert len(series) == 2
        assert series[0]["series_id"] == "GDP"
        assert series[0]["title"] == "Gross Domestic Product"
        assert series[0]["frequency"] == "Quarterly"
        assert series[0]["source"] == "fred"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"FRED_API_KEY": "test-fred-key"})
    async def test_observations(self):
        http = _mock_http({
            "series/observations": _response(json_data=FRED_OBS_JSON),
        })
        result = await _fred_search(
            "https://api.stlouisfed.org/fred",
            {"series_id": "GDP"},
            http,
        )
        assert result["series_id"] == "GDP"
        assert len(result["observations"]) == 2
        assert result["observations"][0]["date"] == "2025-01-01"
        assert result["source"] == "fred"

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        http = _mock_http({})
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FRED_API_KEY", None)
            with pytest.raises(ValueError, match="FRED_API_KEY"):
                await _fred_search(
                    "https://api.stlouisfed.org/fred",
                    {"query": "GDP"},
                    http,
                )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"FRED_API_KEY": "test-fred-key"})
    async def test_empty_results(self):
        http = _mock_http({
            "series/search": _response(json_data={"seriess": []}),
        })
        result = await _fred_search(
            "https://api.stlouisfed.org/fred",
            {"query": "zzzzz"},
            http,
        )
        assert result["series"] == []


# ---------------------------------------------------------------------------
# World Bank tests
# ---------------------------------------------------------------------------

class TestWorldBankFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "indicator": _response(json_data=WB_INDICATOR_SEARCH_JSON),
        })
        result = await _world_bank_search(
            "https://api.worldbank.org/v2",
            {"query": "GDP"},
            http,
        )
        indicators = result["indicators"]
        assert len(indicators) == 1  # Only "GDP (current US$)" matches
        assert indicators[0]["indicator_id"] == "NY.GDP.MKTP.CD"
        assert indicators[0]["source"] == "world-bank"

    @pytest.mark.asyncio
    async def test_indicator_data(self):
        http = _mock_http({
            "indicator/NY.GDP.MKTP.CD": _response(json_data=WB_DATA_JSON),
        })
        result = await _world_bank_search(
            "https://api.worldbank.org/v2",
            {"indicator": "NY.GDP.MKTP.CD", "country": "US"},
            http,
        )
        records = result["indicators"]
        assert len(records) == 2
        assert records[0]["country"] == "United States"
        assert records[0]["year"] == 2023
        assert records[0]["source"] == "world-bank"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "indicator": _response(json_data=[{"page": 1}, None]),
        })
        result = await _world_bank_search(
            "https://api.worldbank.org/v2",
            {"query": "zzzzz"},
            http,
        )
        assert result["indicators"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        data = [{"page": 1}, [{"id": "XX.1", "name": "Test metric XX"}]]
        http = _mock_http({
            "indicator": _response(json_data=data),
        })
        result = await _world_bank_search(
            "https://api.worldbank.org/v2",
            {"query": "XX"},
            http,
        )
        assert result["indicators"][0]["indicator_id"] == "XX.1"


# ---------------------------------------------------------------------------
# BLS tests
# ---------------------------------------------------------------------------

class TestBlsFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "timeseries/data": _response(json_data=BLS_JSON),
        })
        result = await _bls_search(
            "https://api.bls.gov/publicAPI/v2",
            {"series_ids": ["CES0000000001"]},
            http,
        )
        series = result["series"]
        assert len(series) == 1
        assert series[0]["series_id"] == "CES0000000001"
        assert len(series[0]["observations"]) == 2
        assert series[0]["observations"][0]["year"] == "2025"
        assert series[0]["source"] == "bls"

    @pytest.mark.asyncio
    async def test_no_series_ids(self):
        """Query without series IDs returns a helpful note."""
        http = _mock_http({})
        result = await _bls_search(
            "https://api.bls.gov/publicAPI/v2",
            {"query": "unemployment"},
            http,
        )
        assert result["series"] == []
        assert "note" in result
        assert "series IDs" in result["note"]

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "timeseries/data": _response(json_data={"Results": {"series": []}}),
        })
        result = await _bls_search(
            "https://api.bls.gov/publicAPI/v2",
            {"series_ids": ["INVALID"]},
            http,
        )
        assert result["series"] == []


# ---------------------------------------------------------------------------
# SEC EDGAR tests
# ---------------------------------------------------------------------------

class TestSecEdgarFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http({
            "search-index": _response(json_data=EDGAR_JSON),
        })
        result = await _sec_edgar_search(
            "https://efts.sec.gov/LATEST",
            {"query": "Apple"},
            http,
        )
        filings = result["filings"]
        assert len(filings) == 2
        assert filings[0]["company_name"] == "Apple Inc."
        assert filings[0]["form_type"] == "10-K"
        assert filings[0]["filing_date"] == "2025-01-15"
        assert filings[0]["source"] == "sec-edgar"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http({
            "search-index": _response(json_data={"hits": {"hits": []}}, status=200),
        })
        result = await _sec_edgar_search(
            "https://efts.sec.gov/LATEST",
            {"query": "zzzzz"},
            http,
        )
        assert result["filings"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Entry with fallback field names (entity_name, file_type, etc.)."""
        http = _mock_http({
            "search-index": _response(json_data=EDGAR_JSON),
        })
        result = await _sec_edgar_search(
            "https://efts.sec.gov/LATEST",
            {"query": "Google"},
            http,
        )
        # Second filing uses fallback field names
        assert result["filings"][1]["company_name"] == "Google LLC"
        assert result["filings"][1]["form_type"] == "10-Q"
