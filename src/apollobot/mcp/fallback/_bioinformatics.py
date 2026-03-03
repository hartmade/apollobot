"""
Fallback adapters for bioinformatics servers:
GEO, GenBank, UniProt, Ensembl, KEGG, PDB.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry, post_with_retry, safe_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GEO (Gene Expression Omnibus) — via NCBI E-utils
# ---------------------------------------------------------------------------

async def _geo_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search GEO datasets via NCBI E-utilities (esearch → esummary)."""
    query = extract_query(params, "organism")
    limit = min(params.get("limit", params.get("retmax", params.get("max_results", 20))), 100)

    # Step 1: esearch to get GDS IDs
    search_resp = await http.get(
        f"{api_base}/esearch.fcgi",
        params={
            "db": "gds",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        },
    )
    search_resp.raise_for_status()
    id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not id_list:
        logger.info("GEO fallback: 0 datasets for query=%r", query)
        return {"datasets": []}

    # Step 2: esummary to get dataset metadata
    summary_resp = await http.get(
        f"{api_base}/esummary.fcgi",
        params={
            "db": "gds",
            "id": ",".join(id_list),
            "retmode": "json",
        },
    )
    summary_resp.raise_for_status()
    result = summary_resp.json().get("result", {})

    datasets = []
    for uid in id_list:
        doc = result.get(uid, {})
        if not doc:
            continue
        datasets.append({
            "accession": doc.get("accession", ""),
            "title": doc.get("title", ""),
            "summary": doc.get("summary", ""),
            "organism": doc.get("taxon", ""),
            "platform": doc.get("gpl", ""),
            "sample_count": safe_int(doc.get("n_samples", 0)),
            "source": "geo",
        })

    logger.info("GEO fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# GenBank — via NCBI E-utils
# ---------------------------------------------------------------------------

async def _genbank_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search GenBank nucleotide sequences via NCBI E-utilities."""
    query = extract_query(params, "gene", "accession")
    limit = min(params.get("limit", params.get("retmax", params.get("max_results", 20))), 100)

    # Step 1: esearch
    search_resp = await http.get(
        f"{api_base}/esearch.fcgi",
        params={
            "db": "nucleotide",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        },
    )
    search_resp.raise_for_status()
    id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

    if not id_list:
        logger.info("GenBank fallback: 0 sequences for query=%r", query)
        return {"sequences": []}

    # Step 2: esummary
    summary_resp = await http.get(
        f"{api_base}/esummary.fcgi",
        params={
            "db": "nucleotide",
            "id": ",".join(id_list),
            "retmode": "json",
        },
    )
    summary_resp.raise_for_status()
    result = summary_resp.json().get("result", {})

    sequences = []
    for uid in id_list:
        doc = result.get(uid, {})
        if not doc:
            continue
        sequences.append({
            "accession": doc.get("caption", doc.get("accessionversion", "")),
            "title": doc.get("title", ""),
            "organism": doc.get("organism", ""),
            "length": safe_int(doc.get("slen", 0)),
            "moltype": doc.get("moltype", ""),
            "source": "genbank",
        })

    logger.info("GenBank fallback: %d sequences for query=%r", len(sequences), query)
    return {"sequences": sequences}


# ---------------------------------------------------------------------------
# UniProt — REST API
# ---------------------------------------------------------------------------

async def _uniprot_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search UniProt proteins via the REST API."""
    query = extract_query(params, "gene", "protein", "protein_name", "accession")
    limit = min(params.get("limit", params.get("size", 20)), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/uniprotkb/search",
        params={
            "query": query,
            "format": "json",
            "size": limit,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    proteins = []
    for entry in data.get("results", []):
        # Protein name: primary recommended name
        prot_desc = entry.get("proteinDescription", {})
        rec_name = prot_desc.get("recommendedName", {})
        protein_name = rec_name.get("fullName", {}).get("value", "")
        if not protein_name:
            # Fall back to submissionNames
            sub_names = prot_desc.get("submissionNames", [])
            if sub_names:
                protein_name = sub_names[0].get("fullName", {}).get("value", "")

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "")

        # Gene names
        gene_names = []
        for gene in entry.get("genes", []):
            name = gene.get("geneName", {}).get("value", "")
            if name:
                gene_names.append(name)

        proteins.append({
            "accession": entry.get("primaryAccession", ""),
            "protein_name": protein_name,
            "organism": organism,
            "gene_names": gene_names,
            "length": safe_int(entry.get("sequence", {}).get("length", 0)),
            "source": "uniprot",
        })

    logger.info("UniProt fallback: %d proteins for query=%r", len(proteins), query)
    return {"proteins": proteins}


# ---------------------------------------------------------------------------
# Ensembl — REST API
# ---------------------------------------------------------------------------

async def _ensembl_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Ensembl genes by symbol via the REST API."""
    query = extract_query(params, "gene", "symbol", "gene_name")
    species = params.get("species", "homo_sapiens")

    resp = await get_with_retry(
        http,
        f"{api_base}/xrefs/symbol/{species}/{query}",
        headers={"Content-Type": "application/json"},
    )
    if resp.status_code == 400:
        # Symbol not found
        logger.info("Ensembl fallback: no genes for query=%r", query)
        return {"genes": []}
    resp.raise_for_status()
    data = resp.json()

    genes = []
    seen_ids: set[str] = set()
    for xref in data:
        ens_id = xref.get("id", "")
        if not ens_id or ens_id in seen_ids:
            continue
        seen_ids.add(ens_id)
        genes.append({
            "id": ens_id,
            "display_name": query,
            "description": xref.get("description", ""),
            "biotype": xref.get("type", ""),
            "species": species,
            "source": "ensembl",
        })

    logger.info("Ensembl fallback: %d genes for query=%r", len(genes), query)
    return {"genes": genes}


# ---------------------------------------------------------------------------
# KEGG — REST API (tab-delimited text)
# ---------------------------------------------------------------------------

async def _kegg_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search KEGG entries via the REST API."""
    database = params.get("database", "pathway")

    # If the LLM provided a pathway_list, fetch each pathway's info via /get.
    # Check this BEFORE extract_query to avoid the list being stringified.
    pathway_list = params.get("pathway_list", params.get("pathways", []))
    if isinstance(pathway_list, str):
        pathway_list = [pathway_list]
    if pathway_list:
        entries = []
        for pid in pathway_list:
            resp = await http.get(f"{api_base}/get/{pid}")
            if resp.status_code != 200:
                continue
            # First line of KEGG flat file contains the entry name
            lines = resp.text.split("\n")
            name = ""
            for line in lines:
                if line.startswith("NAME"):
                    name = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                    break
            entries.append({
                "id": pid,
                "description": name,
                "source": "kegg",
            })
        logger.info("KEGG fallback: %d entries from pathway_list", len(entries))
        return {"entries": entries}

    # Text search fallback
    query = extract_query(params, "pathway", "compound")
    if not query:
        logger.info("KEGG fallback: no query extracted from params=%r", list(params.keys()))
        return {"entries": []}

    resp = await http.get(f"{api_base}/find/{database}/{query}")
    if resp.status_code != 200:
        logger.info("KEGG fallback: no results for query=%r (status %d)", query, resp.status_code)
        return {"entries": []}

    entries = []
    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        entry_id = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
        entries.append({
            "id": entry_id,
            "description": description,
            "source": "kegg",
        })

    logger.info("KEGG fallback: %d entries for query=%r", len(entries), query)
    return {"entries": entries}


# ---------------------------------------------------------------------------
# PDB (Protein Data Bank) — RCSB REST API
# ---------------------------------------------------------------------------

async def _pdb_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search PDB structures via the RCSB search and data APIs."""
    query = extract_query(params, "structure", "protein", "pdb_id")
    limit = min(params.get("limit", 10), 50)

    # Step 1: full-text search via RCSB search API
    search_payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    search_resp = await post_with_retry(
        http,
        "https://search.rcsb.org/rcsbsearch/v2/query",
        json=search_payload,
    )
    if search_resp.status_code != 200:
        logger.info("PDB fallback: search returned status %d", search_resp.status_code)
        return {"structures": []}

    search_data = search_resp.json()
    result_set = search_data.get("result_set", [])
    pdb_ids = [r.get("identifier", "") for r in result_set if r.get("identifier")]

    if not pdb_ids:
        logger.info("PDB fallback: 0 structures for query=%r", query)
        return {"structures": []}

    # Step 2: fetch metadata for each PDB ID
    structures = []
    for pdb_id in pdb_ids[:limit]:
        entry_resp = await http.get(f"{api_base}/core/entry/{pdb_id}")
        if entry_resp.status_code != 200:
            continue
        entry = entry_resp.json()

        # Extract organism from entity source
        organism = ""
        sources = entry.get("rcsb_entry_info", {})

        # Title and method
        struct = entry.get("struct", {})
        exptl = entry.get("exptl", [{}])

        # Resolution
        reflns = entry.get("reflns", [{}])
        resolution = None
        if reflns:
            resolution = reflns[0].get("d_resolution_high")
        if resolution is None:
            resolution = entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])
            if isinstance(resolution, list) and resolution:
                resolution = resolution[0]

        structures.append({
            "pdb_id": pdb_id,
            "title": struct.get("title", ""),
            "organism": organism,
            "method": exptl[0].get("method", "") if exptl else "",
            "resolution": resolution,
            "source": "pdb",
        })

    logger.info("PDB fallback: %d structures for query=%r", len(structures), query)
    return {"structures": structures}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "geo": _geo_search,
    "genbank": _genbank_search,
    "uniprot": _uniprot_search,
    "ensembl": _ensembl_search,
    "kegg": _kegg_search,
    "pdb": _pdb_search,
}
