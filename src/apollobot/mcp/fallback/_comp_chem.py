"""
Fallback adapters for computational chemistry servers:
PubChem, ChEMBL, AlphaFold DB, ZINC.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry, safe_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PubChem — PUG REST
# ---------------------------------------------------------------------------

def _extract_pubchem_props(pc: dict[str, Any]) -> dict[str, Any]:
    """Extract key properties from a PC_Compound record."""
    props: dict[str, Any] = {}
    for prop in pc.get("props", []):
        urn = prop.get("urn", {})
        label = urn.get("label", "")
        name = urn.get("name", "")
        val = prop.get("value", {})
        value = val.get("sval", val.get("fval", val.get("ival")))
        if label == "IUPAC Name" and name == "Preferred":
            props["name"] = value
        elif label == "Molecular Formula":
            props["formula"] = value
        elif label == "Molecular Weight":
            props["molecular_weight"] = value
        elif label == "SMILES" and name == "Canonical":
            props["smiles"] = value
    return props


async def _pubchem_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search PubChem compounds by name via PUG REST."""
    limit = min(params.get("limit", 5), 20)

    # If LLM provided a list of compound names, search for each.
    # The LLM may use "compound_names", "names", or "compounds" as the key.
    compound_names = params.get("compound_names", params.get("names", params.get("compounds", [])))
    if isinstance(compound_names, str):
        compound_names = [compound_names]
    if compound_names:
        all_compounds: list[dict[str, Any]] = []
        for name in compound_names[:limit]:
            resp = await get_with_retry(http, f"{api_base}/compound/name/{name}/JSON")
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            for pc in data.get("PC_Compounds", [])[:1]:
                cid = safe_int(pc.get("id", {}).get("id", {}).get("cid", 0))
                props = _extract_pubchem_props(pc)
                all_compounds.append({
                    "cid": cid,
                    "name": props.get("name", name),
                    "formula": props.get("formula", ""),
                    "molecular_weight": props.get("molecular_weight"),
                    "smiles": props.get("smiles", ""),
                    "source": "pubchem",
                })
        logger.info("PubChem fallback: %d compounds from compound_names list", len(all_compounds))
        return {"compounds": all_compounds}

    # Single compound search
    query = extract_query(params, "compound_name", "name", "compound")
    if not query:
        logger.info("PubChem fallback: no query extracted from params=%r", list(params.keys()))
        return {"compounds": []}

    # PUG REST: /compound/name/{name}/JSON returns matching compound(s)
    resp = await get_with_retry(
        http,
        f"{api_base}/compound/name/{query}/JSON",
    )
    if resp.status_code == 404:
        logger.info("PubChem fallback: no compounds for query=%r", query)
        return {"compounds": []}
    resp.raise_for_status()
    data = resp.json()

    compounds = []
    for pc in data.get("PC_Compounds", [])[:limit]:
        cid = safe_int(pc.get("id", {}).get("id", {}).get("cid", 0))
        props = _extract_pubchem_props(pc)

        compounds.append({
            "cid": cid,
            "name": props.get("name", query),
            "formula": props.get("formula", ""),
            "molecular_weight": props.get("molecular_weight"),
            "smiles": props.get("smiles", ""),
            "source": "pubchem",
        })

    logger.info("PubChem fallback: %d compounds for query=%r", len(compounds), query)
    return {"compounds": compounds}


# ---------------------------------------------------------------------------
# ChEMBL — EBI REST API
# ---------------------------------------------------------------------------

async def _chembl_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search ChEMBL molecules via the EBI REST API."""
    query = extract_query(params, "molecule", "compound", "drug")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/molecule/search",
        params={"q": query, "limit": limit, "format": "json"},
    )
    if resp.status_code == 404:
        logger.info("ChEMBL fallback: no molecules for query=%r", query)
        return {"molecules": []}
    resp.raise_for_status()
    data = resp.json()

    molecules = []
    for mol in data.get("molecules", []):
        molecules.append({
            "chembl_id": mol.get("molecule_chembl_id", ""),
            "pref_name": mol.get("pref_name", ""),
            "molecule_type": mol.get("molecule_type", ""),
            "max_phase": safe_int(mol.get("max_phase", 0)),
            "source": "chembl",
        })

    logger.info("ChEMBL fallback: %d molecules for query=%r", len(molecules), query)
    return {"molecules": molecules}


# ---------------------------------------------------------------------------
# AlphaFold DB — EBI API
# ---------------------------------------------------------------------------

async def _alphafold_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Look up AlphaFold structure predictions by UniProt accession."""
    accession = extract_query(params, "accession", "uniprot_id", "uniprot_accession")
    if not accession:
        return {"predictions": []}

    # AlphaFold API expects a UniProt accession (e.g. "P04637"), not free-text
    # search.  Accessions are alphanumeric, 6-10 chars, no spaces.  If the
    # query looks like natural language, return empty to avoid 400 errors.
    if " " in accession or len(accession) > 20:
        logger.info("AlphaFold fallback: query %r looks like text, not an accession — skipping", accession[:40])
        return {"predictions": []}

    resp = await get_with_retry(
        http,
        f"{api_base}/prediction/{accession}",
    )
    if resp.status_code == 404:
        logger.info("AlphaFold fallback: no prediction for accession=%r", accession)
        return {"predictions": []}
    resp.raise_for_status()

    data = resp.json()
    # API may return a list or single object
    entries = data if isinstance(data, list) else [data]

    predictions = []
    for entry in entries:
        predictions.append({
            "entry_id": entry.get("entryId", ""),
            "uniprot_accession": entry.get("uniprotAccession", accession),
            "gene": entry.get("gene", ""),
            "organism": entry.get("organismScientificName", ""),
            "confidence": entry.get("globalMetricValue"),
            "source": "alphafold-db",
        })

    logger.info("AlphaFold fallback: %d predictions for accession=%r", len(predictions), accession)
    return {"predictions": predictions}


# ---------------------------------------------------------------------------
# ZINC — docking.org API
# ---------------------------------------------------------------------------

async def _zinc_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search ZINC commercially available compounds."""
    query = extract_query(params, "compound", "smiles", "name")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/substances/search",
        params={"q": query, "count": limit},
    )
    if resp.status_code != 200:
        logger.info("ZINC fallback: no results (status %d)", resp.status_code)
        return {"substances": []}

    data = resp.json()
    items = data if isinstance(data, list) else data.get("substances", data.get("results", []))

    substances = []
    for item in items[:limit]:
        substances.append({
            "zinc_id": item.get("zinc_id", item.get("id", "")),
            "smiles": item.get("smiles", ""),
            "mwt": item.get("mwt"),
            "logp": item.get("logp"),
            "source": "zinc",
        })

    logger.info("ZINC fallback: %d substances for query=%r", len(substances), query)
    return {"substances": substances}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "pubchem": _pubchem_search,
    "chembl": _chembl_search,
    "alphafold-db": _alphafold_search,
    "zinc": _zinc_search,
}
