# ApolloBot

[![PyPI version](https://badge.fury.io/py/apollobot.svg)](https://badge.fury.io/py/apollobot)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**The autonomous research engine by [Frontier Science](https://frontierscience.ai). Give it a question. Get back a paper.**

ApolloBot is an open-source framework that connects frontier AI models to scientific data sources, compute resources, and analysis tools via [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) — enabling fully autonomous computational research.

```bash
pip install apollobot
apollo init
apollo research "Does gut microbiome diversity correlate with epigenetic age acceleration?"
```

That's it. The agent handles literature review, data acquisition, analysis, and manuscript drafting. You handle the science — reviewing, steering, and publishing.

---

## What It Does

ApolloBot is an **autonomous research agent** that:

1. **Parses** a natural language research objective into a structured research plan
2. **Reviews** existing literature via PubMed, arXiv, Semantic Scholar
3. **Acquires** data from public repositories (GEO, GenBank, FRED, HuggingFace, etc.)
4. **Analyzes** data using domain-appropriate statistical and computational methods
5. **Generates** publication-ready manuscripts with figures, tables, and full provenance logs
6. **Self-reviews** its own work for statistical validity and methodological soundness

All via MCP servers — modular, extensible, and transparent.

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for at least one frontier AI provider (Anthropic recommended)

### Install

```bash
pip install apollobot
```

### Initialize

```bash
apollo init
```

This walks you through:
- Your identity (name, affiliation, ORCID)
- Your research domain (bioinformatics, physics, CS/ML, comp chem, economics)
- API key configuration
- Compute preferences (local, cloud, hybrid)

### Run Your First Research Session

```bash
# Simple — one line
apollo research "What is the relationship between telomere length and DNA methylation age in publicly available cohort data?"

# Structured — from a mission file
apollo research --from mission.yaml

# Specific mode
apollo research --mode meta-analysis "CRISPR off-target effects in therapeutic applications"
```

### Monitor Progress

```bash
# Check session status
apollo status

# List all sessions
apollo list
```

## Research Modes

| Mode | Use Case | Description |
|------|----------|-------------|
| `hypothesis` | Testing specific claims | Classical scientific method. Attempts to falsify. |
| `exploratory` | Pattern discovery | Data-mining with built-in multiple comparison correction. |
| `meta-analysis` | Literature synthesis | Systematic review across hundreds of papers. |
| `replication` | Reproducing studies | Adversarial replication of existing published work. |
| `simulation` | Theoretical exploration | Build and run computational models and simulations. |

```bash
apollo research --mode replication --paper arxiv:2401.12345
apollo research --mode exploratory --dataset GSE184571
apollo research --mode simulation "Agent-based model of monetary policy transmission"
```

## MCP Architecture

Every capability is an MCP server. ApolloBot ships with connectors for:

**Data Sources**
- PubMed / PMC (biomedical literature)
- arXiv (preprints)
- Semantic Scholar (citation graphs)
- GEO / GenBank / UniProt (genomics)
- FRED / World Bank (economics)
- HuggingFace Datasets (ML)
- Materials Project (physics/materials)
- PubChem / ChEMBL (chemistry)

**Compute**
- Local execution (NumPy, SciPy, scikit-learn, statsmodels)
- GPU provisioning (Lambda Labs, AWS, RunPod)
- Simulation engines (configurable per domain)

**Writing**
- LaTeX manuscript generation
- Figure and table creation (matplotlib, seaborn, plotly)
- Citation management (BibTeX)

### Adding Custom MCP Servers

```python
# Register a custom data source
from apollobot.mcp import register_server

register_server(
    name="my-lab-database",
    url="http://localhost:8080/mcp",
    description="Internal proteomics database",
    domain="bioinformatics"
)
```

Or via config:

```yaml
# ~/.apollobot/servers.yaml
custom_servers:
  - name: university-hpc
    url: https://hpc.myuniversity.edu/mcp
    auth: bearer
    token_env: UNI_HPC_TOKEN
```

## Research Output

Every completed session produces:

```
~/apollobot-research/session-001/
├── manuscript.tex              # Full paper, journal-formatted
├── manuscript.pdf              # Compiled PDF
├── figures/                    # Publication-quality figures
├── data/                       # Processed datasets
│   ├── raw/                    # Original downloaded data
│   └── processed/              # Cleaned and transformed
├── analysis/                   # All executed code
│   ├── scripts/                # Analysis scripts
│   └── notebooks/              # Jupyter notebooks (optional)
├── provenance/
│   ├── execution_log.json      # Every decision and action
│   ├── data_lineage.json       # Data transformation chain
│   └── model_calls.json        # All LLM interactions
├── review/
│   ├── self_review.md          # AI self-review report
│   └── statistical_audit.json  # Automated stats checking
├── replication_kit/
│   ├── environment.yml         # Conda environment spec
│   ├── replicate.sh            # One-command reproduction
│   └── checksums.sha256        # Data integrity hashes
└── mission.yaml                # Original research objective
```

## Publishing to Frontier Science Journal

ApolloBot integrates directly with [Frontier Science Journal](https://frontierscience.ai/journal) — the open-access journal purpose-built for AI-assisted research.

```bash
# Submit directly from a completed session
apollo submit --session session-001 --journal frontier

# Check submission status
apollo submit --status
```

Frontier Science Journal accepts submissions from any source, but ApolloBot-produced papers include full provenance logs that streamline the review process.

## Mission Files

For complex research objectives, define a mission file:

```yaml
# mission.yaml
title: "Microplastic Epigenetic Effects"

objective: >
  Investigate whether chronic microplastic exposure induces
  heritable DNA methylation changes in aquatic organisms

hypotheses:
  - Microplastic exposure alters methylation at stress-response loci
  - Changes persist in F1 generation without continued exposure
  - Effect size correlates with particle concentration

mode: hypothesis

constraints:
  compute_budget: 50.00        # USD
  time_limit: 48h
  data_sources: public_only
  ethics: observational_only

domain: bioinformatics
resource_pack: bioinformatics  # auto-connect domain MCP servers

checkpoints:
  - after: literature_review
    action: require_approval
  - after: data_acquisition
    action: notify
  - after: analysis
    action: require_approval

output:
  format: paper_draft
  target_journal: frontier
  include_provenance: true
  generate_notebooks: true
```

## Research Integrity

ApolloBot is built with scientific rigor as a core design principle:

- **Anti-confirmation bias**: Hypothesis mode actively seeks disconfirming evidence
- **Multiple comparison correction**: Exploratory mode applies Bonferroni/FDR by default
- **Effect size reporting**: Always reported alongside p-values
- **Full provenance**: Every LLM call, data transformation, and decision is logged
- **Reproducibility**: Replication kits generated automatically
- **Self-review**: Built-in statistical and methodological auditing before output

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas:
- **Domain packs**: Add MCP server connectors for new data sources
- **Research modes**: Implement new research methodologies
- **Analysis methods**: Add statistical and computational tools
- **Review checks**: Improve the self-review engine

## Roadmap

- [x] Core agent loop and research modes
- [x] Bioinformatics domain pack
- [x] Computational physics domain pack
- [x] CS/ML domain pack
- [x] Computational chemistry domain pack
- [x] Quantitative economics domain pack
- [ ] Web dashboard UI
- [ ] Multi-agent collaborative research
- [ ] Robotic lab integration (MCP → physical instruments)
- [ ] Enterprise deployment (private MCP, compliance, audit)

## License

Apache 2.0 — use it freely, commercially or academically.

## Community

- [GitHub Issues](https://github.com/frontier-science/apollobot/issues)
- [Frontier Science Journal](https://frontierscience.ai/journal)
- [Documentation](https://apollobot.dev)

---

*Part of [Frontier Science](https://frontierscience.ai) — the future of knowledge creation.*
