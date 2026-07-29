"""
Built-in MCP server definitions for each domain.

These are the default data source and tool connectors that ship with ApolloBot.
Each domain pack registers a curated set of MCP servers that the agent
can use immediately after setup.

In v1, many of these are thin wrappers around public REST APIs.
The MCP protocol layer normalizes them into a consistent interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass
class BuiltinServer:
    """Definition of a built-in MCP server."""

    name: str
    proxy_path: str
    description: str
    domain: str
    category: str  # data, literature, compute, analysis
    api_base: str = ""  # the underlying public API
    requires_key: bool = False
    key_env_var: str = ""

    @property
    def url(self) -> str:
        """Configured adapter endpoint, or empty for direct public-API mode."""
        return resolve_builtin_proxy_url(self)


def resolve_builtin_proxy_url(
    server: BuiltinServer,
    proxy_base: str | None = None,
) -> str:
    """Resolve a built-in connector against an optional deployed MCP proxy.

    ApolloBot uses the audited direct adapters when no proxy base is configured.
    This avoids treating a speculative or undeployed hostname as a live service.
    """
    configured = (
        os.getenv("APOLLOBOT_MCP_PROXY_URL", "") if proxy_base is None else proxy_base
    ).strip()
    if not configured:
        return ""

    parsed = urlsplit(configured)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "APOLLOBOT_MCP_PROXY_URL must be an absolute HTTP(S) URL without "
            "credentials, query parameters, or a fragment"
        )
    return f"{configured.rstrip('/')}/{server.proxy_path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Shared / cross-domain servers
# ---------------------------------------------------------------------------

LITERATURE_SERVERS = [
    BuiltinServer(
        name="pubmed",
        proxy_path="pubmed",
        description="Search and retrieve biomedical literature from PubMed/MEDLINE",
        domain="shared",
        category="literature",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="arxiv",
        proxy_path="arxiv",
        description="Search and retrieve preprints from arXiv",
        domain="shared",
        category="literature",
        api_base="https://export.arxiv.org/api",
    ),
    BuiltinServer(
        name="semantic-scholar",
        proxy_path="semantic-scholar",
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
        proxy_path="geo",
        description="Gene Expression Omnibus — microarray and seq datasets",
        domain="bioinformatics",
        category="data",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="genbank",
        proxy_path="genbank",
        description="GenBank nucleotide sequence database",
        domain="bioinformatics",
        category="data",
        api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    ),
    BuiltinServer(
        name="uniprot",
        proxy_path="uniprot",
        description="Universal Protein Resource — protein sequences and annotation",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.uniprot.org",
    ),
    BuiltinServer(
        name="ensembl",
        proxy_path="ensembl",
        description="Ensembl genome browser REST API",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.ensembl.org",
    ),
    BuiltinServer(
        name="kegg",
        proxy_path="kegg",
        description="KEGG pathway and molecular interaction databases",
        domain="bioinformatics",
        category="data",
        api_base="https://rest.kegg.jp",
    ),
    BuiltinServer(
        name="pdb",
        proxy_path="pdb",
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
        proxy_path="materials-project",
        description="Materials Project — computed materials properties",
        domain="physics",
        category="data",
        api_base="https://api.materialsproject.org",
        requires_key=True,
        key_env_var="MP_API_KEY",
    ),
    BuiltinServer(
        name="nist",
        proxy_path="nist",
        description="NIST physical and chemical reference data",
        domain="physics",
        category="data",
        api_base="https://physics.nist.gov/cgi-bin/cuu",
    ),
    BuiltinServer(
        name="cern-opendata",
        proxy_path="cern-opendata",
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
        proxy_path="huggingface",
        description="HuggingFace — models, datasets, and spaces",
        domain="cs_ml",
        category="data",
        api_base="https://huggingface.co/api",
        requires_key=True,
        key_env_var="HF_TOKEN",
    ),
    BuiltinServer(
        name="papers-with-code",
        proxy_path="pwc",
        description="Papers With Code — benchmarks, methods, and results",
        domain="cs_ml",
        category="literature",
        api_base="https://paperswithcode.com/api/v1",
    ),
    BuiltinServer(
        name="openml",
        proxy_path="openml",
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
        proxy_path="pubchem",
        description="PubChem — chemical structures, properties, and bioactivities",
        domain="comp_chem",
        category="data",
        api_base="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    ),
    BuiltinServer(
        name="chembl",
        proxy_path="chembl",
        description="ChEMBL — bioactivity data for drug-like molecules",
        domain="comp_chem",
        category="data",
        api_base="https://www.ebi.ac.uk/chembl/api/data",
    ),
    BuiltinServer(
        name="alphafold-db",
        proxy_path="alphafold",
        description="AlphaFold Protein Structure Database",
        domain="comp_chem",
        category="data",
        api_base="https://alphafold.ebi.ac.uk/api",
    ),
    BuiltinServer(
        name="zinc",
        proxy_path="zinc",
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
        proxy_path="fred",
        description="Federal Reserve Economic Data",
        domain="economics",
        category="data",
        api_base="https://api.stlouisfed.org/fred",
        requires_key=True,
        key_env_var="FRED_API_KEY",
    ),
    BuiltinServer(
        name="world-bank",
        proxy_path="world-bank",
        description="World Bank Open Data indicators",
        domain="economics",
        category="data",
        api_base="https://api.worldbank.org/v2",
    ),
    BuiltinServer(
        name="bls",
        proxy_path="bls",
        description="Bureau of Labor Statistics public data",
        domain="economics",
        category="data",
        api_base="https://api.bls.gov/publicAPI/v2",
    ),
    BuiltinServer(
        name="sec-edgar",
        proxy_path="sec-edgar",
        description="SEC EDGAR — corporate filings and financial data",
        domain="economics",
        category="data",
        api_base="https://efts.sec.gov/LATEST",
    ),
]

# ---------------------------------------------------------------------------
# Astronomy
# ---------------------------------------------------------------------------

ASTRONOMY_SERVERS = [
    BuiltinServer(
        name="simbad",
        proxy_path="simbad",
        description="SIMBAD astronomical database — stellar and galaxy data",
        domain="astronomy",
        category="data",
        api_base="https://simbad.cds.unistra.fr/simbad/sim-script",
    ),
    BuiltinServer(
        name="mast",
        proxy_path="mast",
        description="MAST — Mikulski Archive for Space Telescopes",
        domain="astronomy",
        category="data",
        api_base="https://mast.stsci.edu/api/v0",
    ),
    BuiltinServer(
        name="vizier",
        proxy_path="vizier",
        description="VizieR — astronomical catalog service",
        domain="astronomy",
        category="data",
        api_base="https://vizier.cds.unistra.fr/viz-bin",
    ),
    BuiltinServer(
        name="nasa-ads",
        proxy_path="nasa-ads",
        description="NASA ADS — astrophysics literature and citations",
        domain="astronomy",
        category="literature",
        api_base="https://api.adsabs.harvard.edu/v1",
        requires_key=True,
        key_env_var="ADS_API_KEY",
    ),
]

# ---------------------------------------------------------------------------
# Climate Science
# ---------------------------------------------------------------------------

CLIMATE_SERVERS = [
    BuiltinServer(
        name="copernicus",
        proxy_path="copernicus",
        description="Copernicus Climate Data Store",
        domain="climate",
        category="data",
        api_base="https://cds.climate.copernicus.eu/api/v2",
    ),
    BuiltinServer(
        name="noaa-giss",
        proxy_path="noaa-giss",
        description="NASA GISS — surface temperature analysis",
        domain="climate",
        category="data",
        api_base="https://data.giss.nasa.gov",
    ),
    BuiltinServer(
        name="noaa-ncei",
        proxy_path="noaa-ncei",
        description="NOAA NCEI — climate and weather datasets",
        domain="climate",
        category="data",
        api_base="https://www.ncei.noaa.gov/cdo-web/api/v2",
        requires_key=True,
        key_env_var="NOAA_API_KEY",
    ),
]

# ---------------------------------------------------------------------------
# Neuroscience
# ---------------------------------------------------------------------------

NEUROSCIENCE_SERVERS = [
    BuiltinServer(
        name="allen-brain",
        proxy_path="allen-brain",
        description="Allen Brain Atlas — gene expression and connectivity",
        domain="neuroscience",
        category="data",
        api_base="https://api.brain-map.org/api/v2",
    ),
    BuiltinServer(
        name="openneuro",
        proxy_path="openneuro",
        description="OpenNeuro — open neuroimaging datasets",
        domain="neuroscience",
        category="data",
        api_base="https://openneuro.org/crn",
    ),
    BuiltinServer(
        name="neurovault",
        proxy_path="neurovault",
        description="NeuroVault — brain imaging results and statistical maps",
        domain="neuroscience",
        category="data",
        api_base="https://neurovault.org/api",
    ),
]

# ---------------------------------------------------------------------------
# Epidemiology
# ---------------------------------------------------------------------------

EPIDEMIOLOGY_SERVERS = [
    BuiltinServer(
        name="owid",
        proxy_path="owid",
        description="Our World in Data — global health and development indicators",
        domain="epidemiology",
        category="data",
        api_base="https://catalog.ourworldindata.org",
    ),
    BuiltinServer(
        name="gho",
        proxy_path="gho",
        description="WHO Global Health Observatory — health statistics",
        domain="epidemiology",
        category="data",
        api_base="https://ghoapi.azureedge.net/api",
    ),
    BuiltinServer(
        name="cdc-wonder",
        proxy_path="cdc-wonder",
        description="CDC WONDER — mortality and population data",
        domain="epidemiology",
        category="data",
        api_base="https://wonder.cdc.gov",
    ),
]

# ---------------------------------------------------------------------------
# Ecology
# ---------------------------------------------------------------------------

ECOLOGY_SERVERS = [
    BuiltinServer(
        name="gbif",
        proxy_path="gbif",
        description="GBIF — global biodiversity occurrence records",
        domain="ecology",
        category="data",
        api_base="https://api.gbif.org/v1",
    ),
    BuiltinServer(
        name="iucn",
        proxy_path="iucn",
        description="IUCN Red List — species conservation status",
        domain="ecology",
        category="data",
        api_base="https://apiv3.iucnredlist.org/api/v3",
        requires_key=True,
        key_env_var="IUCN_API_KEY",
    ),
    BuiltinServer(
        name="obis",
        proxy_path="obis",
        description="OBIS — ocean biodiversity information system",
        domain="ecology",
        category="data",
        api_base="https://api.obis.org/v3",
    ),
]

# ---------------------------------------------------------------------------
# Geology
# ---------------------------------------------------------------------------

GEOLOGY_SERVERS = [
    BuiltinServer(
        name="usgs-earthquake",
        proxy_path="usgs-earthquake",
        description="USGS earthquake catalog — seismic event data",
        domain="geology",
        category="data",
        api_base="https://earthquake.usgs.gov/fdsnws/event/1",
    ),
    BuiltinServer(
        name="iris",
        proxy_path="iris",
        description="IRIS — seismological station and waveform data",
        domain="geology",
        category="data",
        api_base="https://service.iris.edu/fdsnws",
    ),
    BuiltinServer(
        name="pangaea",
        proxy_path="pangaea",
        description="PANGAEA — earth and environmental science data",
        domain="geology",
        category="data",
        api_base="https://ws.pangaea.de/es/dataportal-lgc",
    ),
]

# ---------------------------------------------------------------------------
# Materials Science
# ---------------------------------------------------------------------------

MATERIALS_SERVERS = [
    BuiltinServer(
        name="aflow",
        proxy_path="aflow",
        description="AFLOW — automated computational materials data",
        domain="materials",
        category="data",
        api_base="https://aflow.org/API",
    ),
    BuiltinServer(
        name="nomad",
        proxy_path="nomad",
        description="NOMAD — novel materials discovery archive",
        domain="materials",
        category="data",
        api_base="https://nomad-lab.eu/prod/v1/api/v1",
    ),
    BuiltinServer(
        name="icsd-web",
        proxy_path="icsd",
        description="ICSD — inorganic crystal structure database",
        domain="materials",
        category="data",
        api_base="https://icsd.fiz-karlsruhe.de/api",
    ),
]

# ---------------------------------------------------------------------------
# Psychology
# ---------------------------------------------------------------------------

PSYCHOLOGY_SERVERS = [
    BuiltinServer(
        name="osf",
        proxy_path="osf",
        description="Open Science Framework — preprints, data, and registrations",
        domain="psychology",
        category="data",
        api_base="https://api.osf.io/v2",
    ),
    BuiltinServer(
        name="psychopen",
        proxy_path="psychopen",
        description="PsychArchives — open access psychology research",
        domain="psychology",
        category="data",
        api_base="https://www.psycharchives.org/api",
    ),
    BuiltinServer(
        name="core",
        proxy_path="core",
        description="CORE — open access research paper aggregator",
        domain="psychology",
        category="literature",
        api_base="https://api.core.ac.uk/v3",
        requires_key=True,
        key_env_var="CORE_API_KEY",
    ),
]

# ---------------------------------------------------------------------------
# Mathematics
# ---------------------------------------------------------------------------

MATHEMATICS_SERVERS = [
    BuiltinServer(
        name="oeis",
        proxy_path="oeis",
        description="OEIS — On-Line Encyclopedia of Integer Sequences",
        domain="mathematics",
        category="data",
        api_base="https://oeis.org",
    ),
    BuiltinServer(
        name="zbmath",
        proxy_path="zbmath",
        description="zbMATH Open — mathematical literature and reviews",
        domain="mathematics",
        category="literature",
        api_base="https://api.zbmath.org/v1",
    ),
    BuiltinServer(
        name="crossref-math",
        proxy_path="crossref-math",
        description="Crossref — math-tagged scholarly works and citations",
        domain="mathematics",
        category="literature",
        api_base="https://api.crossref.org/works",
    ),
]

# ---------------------------------------------------------------------------
# Social Science
# ---------------------------------------------------------------------------

SOCIAL_SCIENCE_SERVERS = [
    BuiltinServer(
        name="dataverse",
        proxy_path="dataverse",
        description="Harvard Dataverse — research data repository",
        domain="social_science",
        category="data",
        api_base="https://dataverse.harvard.edu/api",
    ),
    BuiltinServer(
        name="icpsr",
        proxy_path="icpsr",
        description="ICPSR — social science data archive",
        domain="social_science",
        category="data",
        api_base="https://www.icpsr.umich.edu/web/ICPSR/api",
    ),
    BuiltinServer(
        name="census",
        proxy_path="census",
        description="US Census Bureau — demographic and economic data",
        domain="social_science",
        category="data",
        api_base="https://api.census.gov/data",
        requires_key=True,
        key_env_var="CENSUS_API_KEY",
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
    + ASTRONOMY_SERVERS
    + CLIMATE_SERVERS
    + NEUROSCIENCE_SERVERS
    + EPIDEMIOLOGY_SERVERS
    + ECOLOGY_SERVERS
    + GEOLOGY_SERVERS
    + MATERIALS_SERVERS
    + PSYCHOLOGY_SERVERS
    + MATHEMATICS_SERVERS
    + SOCIAL_SCIENCE_SERVERS
)

DOMAIN_PACKS: dict[str, list[BuiltinServer]] = {
    "bioinformatics": LITERATURE_SERVERS + BIOINFORMATICS_SERVERS,
    "physics": LITERATURE_SERVERS + PHYSICS_SERVERS,
    "cs_ml": LITERATURE_SERVERS + CS_ML_SERVERS,
    "comp_chem": LITERATURE_SERVERS + COMP_CHEM_SERVERS,
    "economics": LITERATURE_SERVERS + ECONOMICS_SERVERS,
    "astronomy": LITERATURE_SERVERS + ASTRONOMY_SERVERS,
    "climate": LITERATURE_SERVERS + CLIMATE_SERVERS,
    "neuroscience": LITERATURE_SERVERS + NEUROSCIENCE_SERVERS,
    "epidemiology": LITERATURE_SERVERS + EPIDEMIOLOGY_SERVERS,
    "ecology": LITERATURE_SERVERS + ECOLOGY_SERVERS,
    "geology": LITERATURE_SERVERS + GEOLOGY_SERVERS,
    "materials": LITERATURE_SERVERS + MATERIALS_SERVERS,
    "psychology": LITERATURE_SERVERS + PSYCHOLOGY_SERVERS,
    "mathematics": LITERATURE_SERVERS + MATHEMATICS_SERVERS,
    "social_science": LITERATURE_SERVERS + SOCIAL_SCIENCE_SERVERS,
}


def get_domain_pack(domain: str) -> list[BuiltinServer]:
    """Get the built-in server list for a domain."""
    return DOMAIN_PACKS.get(domain, LITERATURE_SERVERS)
