"""
Built-in MCP server definitions for each domain.

These are the default data source and tool connectors that ship with ApolloBot.
Each domain pack registers a curated set of MCP servers that the agent
can use immediately after setup.

In v1, many of these are thin wrappers around public REST APIs.
The MCP protocol layer normalizes them into a consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuiltinServer:
    """Definition of a built-in MCP server."""

    name: str
    url: str  # in v1, this points to our proxy/adapter service
    description: str
    domain: str
    category: str  # data, literature, compute, analysis
    api_base: str = ""  # the underlying public API
    requires_key: bool = False
    key_env_var: str = ""


# ---------------------------------------------------------------------------
# Shared / cross-domain servers
# ---------------------------------------------------------------------------

LITERATURE_SERVERS = [
    BuiltinServer(
        name="pubmed",
        url="https://mcp.frontierscience.ai/pubmed",
        description="Search and retrieve biomedical literature from PubMed/MEDLINE",
        domain="shared",
        category="literature",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="arxiv",
        url="https://mcp.frontierscience.ai/arxiv",
        description="Search and retrieve preprints from arXiv",
        domain="shared",
        category="literature",
        api_base="http://export.arxiv.org/api",
    ),
    BuiltinServer(
        name="semantic-scholar",
        url="https://mcp.frontierscience.ai/semantic-scholar",
        description="Citation graph search, paper metadata, and recommendations",
        domain="shared",
        category="literature",
        api_base="https://api.semanticscholar.org/graph/v1",
        requires_key=True,
        key_env_var="S2_API_KEY",
    ),
]

# ---------------------------------------------------------------------------
# Bioinformatics
# ---------------------------------------------------------------------------

BIOINFORMATICS_SERVERS = [
    BuiltinServer(
        name="geo",
        url="https://mcp.frontierscience.ai/geo",
        description="Gene Expression Omnibus — microarray and seq datasets",
        domain="bioinformatics",
        category="data",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="genbank",
        url="https://mcp.frontierscience.ai/genbank",
        description="GenBank nucleotide sequence database",
        domain="bioinformatics",
        category="data",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="uniprot",
        url="https://mcp.frontierscience.ai/uniprot",
        description="Universal Protein Resource — protein sequences and annotation",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.uniprot.org",
    ),
    BuiltinServer(
        name="ensembl",
        url="https://mcp.frontierscience.ai/ensembl",
        description="Ensembl genome browser REST API",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.ensembl.org",
    ),
    BuiltinServer(
        name="kegg",
        url="https://mcp.frontierscience.ai/kegg",
        description="KEGG pathway and molecular interaction databases",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.kegg.jp",
    ),
    BuiltinServer(
        name="pdb",
        url="https://mcp.frontierscience.ai/pdb",
        description="Protein Data Bank — 3D structure data",
        domain="bioinformatics",
        category="data",
        api_base="https://data.rcsb.org/rest/v1",
    ),
]

# ---------------------------------------------------------------------------
# Computational Physics
# ---------------------------------------------------------------------------

PHYSICS_SERVERS = [
    BuiltinServer(
        name="materials-project",
        url="https://mcp.frontierscience.ai/materials-project",
        description="Materials Project — computed materials properties",
        domain="physics",
        category="data",
        api_base="https://api.materialsproject.org",
        requires_key=True,
        key_env_var="MP_API_KEY",
    ),
    BuiltinServer(
        name="nist",
        url="https://mcp.frontierscience.ai/nist",
        description="NIST physical and chemical reference data",
        domain="physics",
        category="data",
        api_base="https://physics.nist.gov/cgi-bin/cuu",
    ),
    BuiltinServer(
        name="cern-opendata",
        url="https://mcp.frontierscience.ai/cern-opendata",
        description="CERN Open Data Portal — particle physics datasets",
        domain="physics",
        category="data",
        api_base="https://opendata.cern.ch/api",
    ),
]

# ---------------------------------------------------------------------------
# Computer Science / ML
# ---------------------------------------------------------------------------

CS_ML_SERVERS = [
    BuiltinServer(
        name="huggingface",
        url="https://mcp.frontierscience.ai/huggingface",
        description="HuggingFace — models, datasets, and spaces",
        domain="cs_ml",
        category="data",
        api_base="https://huggingface.co/api",
        requires_key=True,
        key_env_var="HF_TOKEN",
    ),
    BuiltinServer(
        name="papers-with-code",
        url="https://mcp.frontierscience.ai/pwc",
        description="Papers With Code — benchmarks, methods, and results",
        domain="cs_ml",
        category="literature",
        api_base="https://paperswithcode.com/api/v1",
    ),
    BuiltinServer(
        name="openml",
        url="https://mcp.frontierscience.ai/openml",
        description="OpenML — machine learning experiments and datasets",
        domain="cs_ml",
        category="data",
        api_base="https://www.openml.org/api/v1",
    ),
]

# ---------------------------------------------------------------------------
# Computational Chemistry
# ---------------------------------------------------------------------------

COMP_CHEM_SERVERS = [
    BuiltinServer(
        name="pubchem",
        url="https://mcp.frontierscience.ai/pubchem",
        description="PubChem — chemical structures, properties, and bioactivities",
        domain="comp_chem",
        category="data",
        api_base="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    ),
    BuiltinServer(
        name="chembl",
        url="https://mcp.frontierscience.ai/chembl",
        description="ChEMBL — bioactivity data for drug-like molecules",
        domain="comp_chem",
        category="data",
        api_base="https://www.ebi.ac.uk/chembl/api/data",
    ),
    BuiltinServer(
        name="alphafold-db",
        url="https://mcp.frontierscience.ai/alphafold",
        description="AlphaFold Protein Structure Database",
        domain="comp_chem",
        category="data",
        api_base="https://alphafold.ebi.ac.uk/api",
    ),
    BuiltinServer(
        name="zinc",
        url="https://mcp.frontierscience.ai/zinc",
        description="ZINC — commercially available compounds for virtual screening",
        domain="comp_chem",
        category="data",
        api_base="https://zinc.docking.org/api",
    ),
]

# ---------------------------------------------------------------------------
# Quantitative Economics
# ---------------------------------------------------------------------------

ECONOMICS_SERVERS = [
    BuiltinServer(
        name="fred",
        url="https://mcp.frontierscience.ai/fred",
        description="Federal Reserve Economic Data",
        domain="economics",
        category="data",
        api_base="https://api.stlouisfed.org/fred",
        requires_key=True,
        key_env_var="FRED_API_KEY",
    ),
    BuiltinServer(
        name="world-bank",
        url="https://mcp.frontierscience.ai/world-bank",
        description="World Bank Open Data indicators",
        domain="economics",
        category="data",
        api_base="https://api.worldbank.org/v2",
    ),
    BuiltinServer(
        name="bls",
        url="https://mcp.frontierscience.ai/bls",
        description="Bureau of Labor Statistics public data",
        domain="economics",
        category="data",
        api_base="https://api.bls.gov/publicAPI/v2",
    ),
    BuiltinServer(
        name="sec-edgar",
        url="https://mcp.frontierscience.ai/sec-edgar",
        description="SEC EDGAR — corporate filings and financial data",
        domain="economics",
        category="data",
        api_base="https://efts.sec.gov/LATEST",
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_BUILTIN_SERVERS = (
    LITERATURE_SERVERS
    + BIOINFORMATICS_SERVERS
    + PHYSICS_SERVERS
    + CS_ML_SERVERS
    + COMP_CHEM_SERVERS
    + ECONOMICS_SERVERS
)

DOMAIN_PACKS: dict[str, list[BuiltinServer]] = {
    "bioinformatics": LITERATURE_SERVERS + BIOINFORMATICS_SERVERS,
    "physics": LITERATURE_SERVERS + PHYSICS_SERVERS,
    "cs_ml": LITERATURE_SERVERS + CS_ML_SERVERS,
    "comp_chem": LITERATURE_SERVERS + COMP_CHEM_SERVERS,
    "economics": LITERATURE_SERVERS + ECONOMICS_SERVERS,
}


def get_domain_pack(domain: str) -> list[BuiltinServer]:
    """Get the built-in server list for a domain."""
    return DOMAIN_PACKS.get(domain, LITERATURE_SERVERS)
