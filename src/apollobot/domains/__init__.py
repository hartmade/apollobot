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
    "astronomy": DomainPack(
        name="astronomy",
        description="Observational and computational astronomy and astrophysics",
        analysis_methods=[
            "photometry",
            "spectroscopy",
            "astrometry",
            "time_series_analysis",
            "image_stacking",
        ],
        statistical_frameworks=[
            "bayesian_model_selection",
            "periodogram_analysis",
            "aperture_photometry",
            "chi_square_fitting",
        ],
        recommended_packages=[
            "astropy",
            "photutils",
            "scipy",
            "healpy",
        ],
        system_prompt_additions=(
            "Report magnitudes with uncertainties. Use standard astrometric "
            "conventions and coordinate systems. Address systematic uncertainties "
            "from calibration and extinction."
        ),
    ),
    "climate": DomainPack(
        name="climate",
        description="Climate science, atmospheric physics, and earth systems modeling",
        analysis_methods=[
            "climate_projection",
            "reanalysis",
            "trend_detection",
            "spatial_interpolation",
            "ensemble_analysis",
        ],
        statistical_frameworks=[
            "mann_kendall_trend",
            "uncertainty_quantification",
            "extreme_value_theory",
            "spatial_statistics",
        ],
        recommended_packages=[
            "xarray",
            "cartopy",
            "cfgrib",
            "scipy",
        ],
        system_prompt_additions=(
            "Distinguish model projections from observations. Quantify ensemble "
            "spread. Use appropriate baselines for anomaly calculations. Report "
            "spatial and temporal resolution of datasets."
        ),
    ),
    "neuroscience": DomainPack(
        name="neuroscience",
        description="Computational neuroscience and neuroimaging",
        analysis_methods=[
            "fmri_analysis",
            "connectivity_analysis",
            "eeg_processing",
            "neural_decoding",
            "brain_parcellation",
        ],
        statistical_frameworks=[
            "permutation_testing",
            "false_discovery_rate",
            "mixed_effects_models",
            "bayesian_inference",
        ],
        recommended_packages=[
            "nilearn",
            "mne",
            "nibabel",
            "scipy",
        ],
        system_prompt_additions=(
            "Apply appropriate multiple comparison correction for brain imaging. "
            "Report cluster-level statistics. Distinguish activation from "
            "connectivity findings. Address motion artifact concerns."
        ),
    ),
    "epidemiology": DomainPack(
        name="epidemiology",
        description="Epidemiology, public health, and infectious disease modeling",
        analysis_methods=[
            "survival_analysis",
            "outbreak_detection",
            "compartmental_modeling",
            "spatial_epidemiology",
            "meta_analysis",
        ],
        statistical_frameworks=[
            "cox_regression",
            "kaplan_meier",
            "poisson_regression",
            "random_effects_meta_analysis",
        ],
        recommended_packages=[
            "lifelines",
            "statsmodels",
            "geopandas",
            "scipy",
        ],
        system_prompt_additions=(
            "Report incidence and prevalence with confidence intervals. "
            "Address confounding and selection bias. Use appropriate measures "
            "of effect (RR, OR, HR). Consider ecological fallacy in aggregate data."
        ),
    ),
    "ecology": DomainPack(
        name="ecology",
        description="Ecology, biodiversity, and environmental science",
        analysis_methods=[
            "species_distribution_modeling",
            "diversity_indices",
            "community_analysis",
            "population_dynamics",
            "occupancy_modeling",
        ],
        statistical_frameworks=[
            "glmm",
            "rarefaction",
            "ordination",
            "spatial_autocorrelation",
        ],
        recommended_packages=[
            "scikit-bio",
            "geopandas",
            "scipy",
            "statsmodels",
        ],
        system_prompt_additions=(
            "Report biodiversity indices with rarefaction curves. Consider "
            "spatial autocorrelation. Distinguish correlation from causation "
            "in observational field data. Account for detection probability."
        ),
    ),
    "geology": DomainPack(
        name="geology",
        description="Geology, geophysics, and solid earth science",
        analysis_methods=[
            "seismic_analysis",
            "geochemical_modeling",
            "stratigraphic_correlation",
            "geostatistics",
            "spectral_analysis",
        ],
        statistical_frameworks=[
            "kriging",
            "variogram_analysis",
            "uncertainty_propagation",
            "bayesian_geostatistics",
        ],
        recommended_packages=[
            "obspy",
            "pyproj",
            "geopandas",
            "scipy",
        ],
        system_prompt_additions=(
            "Report depths and magnitudes with appropriate precision. "
            "Use standard geological time scales. Consider measurement "
            "uncertainty in field data. Apply proper coordinate transformations."
        ),
    ),
    "materials": DomainPack(
        name="materials",
        description="Materials science, crystallography, and computational materials design",
        analysis_methods=[
            "dft_calculations",
            "crystal_structure_prediction",
            "phase_diagram_analysis",
            "property_screening",
            "defect_analysis",
        ],
        statistical_frameworks=[
            "cross_validation",
            "uncertainty_quantification",
            "pareto_optimization",
            "sensitivity_analysis",
        ],
        recommended_packages=[
            "pymatgen",
            "ase",
            "numpy",
            "scikit-learn",
        ],
        system_prompt_additions=(
            "Report energies in eV/atom. Compare DFT results against "
            "experimental data where available. Address convergence of "
            "computational parameters. Use standard crystallographic notation."
        ),
    ),
    "psychology": DomainPack(
        name="psychology",
        description="Psychology, cognitive science, and behavioral research",
        analysis_methods=[
            "factorial_anova",
            "structural_equation_modeling",
            "item_response_theory",
            "mediation_analysis",
            "meta_analysis",
        ],
        statistical_frameworks=[
            "anova",
            "mixed_effects_models",
            "bayesian_estimation",
            "equivalence_testing",
        ],
        recommended_packages=[
            "pingouin",
            "statsmodels",
            "scipy",
            "factor_analyzer",
        ],
        system_prompt_additions=(
            "Report effect sizes (Cohen's d, eta-squared). Address replication "
            "concerns explicitly. Pre-register analyses where possible. "
            "Distinguish exploratory from confirmatory analyses. Report "
            "power analyses for sample size justification."
        ),
    ),
    "mathematics": DomainPack(
        name="mathematics",
        description="Pure and applied mathematics, combinatorics, and number theory",
        analysis_methods=[
            "numerical_verification",
            "symbolic_computation",
            "conjecture_testing",
            "asymptotic_analysis",
            "computational_enumeration",
        ],
        statistical_frameworks=[
            "monte_carlo_estimation",
            "numerical_integration",
            "convergence_analysis",
            "error_bounds",
        ],
        recommended_packages=[
            "sympy",
            "sage",
            "numpy",
            "mpmath",
        ],
        system_prompt_additions=(
            "Distinguish computational evidence from proof. Report numerical "
            "precision and error bounds explicitly. State conjectures clearly "
            "and separately from theorems. Verify edge cases."
        ),
    ),
    "social_science": DomainPack(
        name="social_science",
        description="Sociology, political science, and quantitative social research",
        analysis_methods=[
            "survey_analysis",
            "network_analysis",
            "content_analysis",
            "causal_inference",
            "multilevel_modeling",
        ],
        statistical_frameworks=[
            "hierarchical_linear_models",
            "propensity_score_matching",
            "instrumental_variables",
            "structural_equation_modeling",
        ],
        recommended_packages=[
            "statsmodels",
            "networkx",
            "pandas",
            "scipy",
        ],
        system_prompt_additions=(
            "Address selection bias and confounding. Report survey design "
            "effects (clustering, stratification, weights). Distinguish "
            "correlation from causation. Consider external validity "
            "and generalizability of findings."
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
