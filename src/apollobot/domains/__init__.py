"""
Domain packs for ApolloBot — domain-specific research configurations.

Each domain pack defines:
- Analysis methods appropriate for the domain
- Statistical frameworks commonly used
- Recommended Python packages
- Domain-specific prompt additions
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainPack:
    """Configuration for a research domain."""

    name: str
    description: str
    analysis_methods: list[str] = field(default_factory=list)
    statistical_frameworks: list[str] = field(default_factory=list)
    recommended_packages: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""


# Default domain packs
_DOMAIN_PACKS: dict[str, DomainPack] = {
    "bioinformatics": DomainPack(
        name="bioinformatics",
        description="Computational biology, genomics, and systems biology",
        analysis_methods=[
            "differential_expression",
            "pathway_analysis",
            "sequence_alignment",
            "phylogenetics",
            "gene_ontology_enrichment",
        ],
        statistical_frameworks=[
            "limma",
            "DESeq2",
            "edgeR",
            "multiple_testing_correction",
        ],
        recommended_packages=[
            "biopython",
            "scanpy",
            "anndata",
            "gseapy",
        ],
        system_prompt_additions=(
            "Focus on biological interpretation. Consider multiple testing "
            "correction for high-dimensional data. Report fold changes and "
            "adjusted p-values."
        ),
    ),
    "physics": DomainPack(
        name="physics",
        description="Computational and theoretical physics",
        analysis_methods=[
            "numerical_simulation",
            "monte_carlo",
            "finite_element",
            "spectral_analysis",
        ],
        statistical_frameworks=[
            "uncertainty_propagation",
            "bayesian_inference",
            "chi_square_fitting",
        ],
        recommended_packages=[
            "numpy",
            "scipy",
            "sympy",
            "astropy",
        ],
        system_prompt_additions=(
            "Report uncertainties and error propagation. Use SI units. "
            "Consider dimensional analysis for sanity checks."
        ),
    ),
    "cs_ml": DomainPack(
        name="cs_ml",
        description="Computer science and machine learning research",
        analysis_methods=[
            "cross_validation",
            "ablation_study",
            "hyperparameter_search",
            "benchmark_evaluation",
        ],
        statistical_frameworks=[
            "bootstrap",
            "significance_testing",
            "confidence_intervals",
        ],
        recommended_packages=[
            "scikit-learn",
            "pytorch",
            "transformers",
            "wandb",
        ],
        system_prompt_additions=(
            "Report standard deviations across runs. Use proper train/val/test "
            "splits. Compare against established baselines."
        ),
    ),
    "comp_chem": DomainPack(
        name="comp_chem",
        description="Computational chemistry and drug discovery",
        analysis_methods=[
            "molecular_docking",
            "qsar_modeling",
            "conformational_analysis",
            "property_prediction",
        ],
        statistical_frameworks=[
            "leave_one_out_cv",
            "external_validation",
            "applicability_domain",
        ],
        recommended_packages=[
            "rdkit",
            "openbabel",
            "mdanalysis",
            "deepchem",
        ],
        system_prompt_additions=(
            "Consider ADMET properties. Report binding affinities with "
            "appropriate units. Validate models on external datasets."
        ),
    ),
    "economics": DomainPack(
        name="economics",
        description="Quantitative economics and econometrics",
        analysis_methods=[
            "regression_analysis",
            "time_series",
            "causal_inference",
            "panel_data",
        ],
        statistical_frameworks=[
            "instrumental_variables",
            "difference_in_differences",
            "regression_discontinuity",
        ],
        recommended_packages=[
            "statsmodels",
            "linearmodels",
            "pandas",
            "arch",
        ],
        system_prompt_additions=(
            "Address endogeneity concerns. Report robust standard errors. "
            "Consider selection bias and confounding."
        ),
    ),
}


def get_domain_pack(domain: str) -> DomainPack:
    """Get the domain pack for a research domain."""
    return _DOMAIN_PACKS.get(
        domain,
        DomainPack(name=domain, description=f"Research in {domain}"),
    )


__all__ = ["DomainPack", "get_domain_pack"]
