"""Pre-balance diagnostic analysis for Ecopath models.

This module provides functions to analyze and visualize model parameters before
balancing, helping identify potential issues with biomasses, vital rates, and
predator-prey relationships.

Based on the Prebal routine by Barbara Bauer (SU, 2016).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core.params import RpathParams


def _calculate_trophic_levels(model: RpathParams) -> pd.Series:
    """Calculate trophic levels for unbalanced model.

    This is a simplified TL calculation for pre-balance diagnostics.
    Uses iterative method based on diet composition.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters

    Returns
    -------
    pd.Series
        Trophic levels indexed by group name
    """
    groups = model.model['Group'].values
    n_groups = len(groups)

    # Initialize trophic levels
    tl = np.ones(n_groups)

    # Set producers (Type=1) to TL=1
    for i, row in model.model.iterrows():
        if row['Type'] == 1:  # Producer
            tl[i] = 1.0

    # Iteratively calculate TL for consumers
    # TL = 1 + weighted average of prey TLs
    max_iterations = 50
    for _iteration in range(max_iterations):
        tl_old = tl.copy()

        for i, group in enumerate(groups):
            group_type = model.model.iloc[i]['Type']

            # Skip producers and detritus
            if group_type in [1, 2]:
                continue

            # Get diet for this consumer
            if group in model.diet.columns:
                diet = model.diet[group]

                # Calculate weighted TL from prey
                prey_tl_sum = 0.0
                diet_sum = 0.0

                for prey_name, diet_frac in diet.items():
                    if diet_frac > 0 and prey_name in groups:
                        prey_idx = np.where(groups == prey_name)[0]
                        if len(prey_idx) > 0:
                            prey_tl_sum += diet_frac * tl_old[prey_idx[0]]
                            diet_sum += diet_frac

                if diet_sum > 0:
                    tl[i] = 1.0 + (prey_tl_sum / diet_sum)

        # Check convergence
        if np.max(np.abs(tl - tl_old)) < 0.001:
            break

    return pd.Series(tl, index=groups, name='TL')


def calculate_biomass_slope(model: RpathParams) -> float:
    """Calculate the biomass decline slope across trophic levels.

    A steep negative slope indicates strong top-down control.
    Typical values: -0.5 to -1.5

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters

    Returns
    -------
    float
        Slope of log10(biomass) vs ordered groups

    Examples
    --------
    >>> slope = calculate_biomass_slope(params)
    >>> print(f"Biomass slope: {slope:.3f}")
    """
    # Get groups with biomass data (exclude detritus and fleets)
    df = model.model[model.model['Type'].isin([0, 1, 2])].copy()

    # Calculate TL if not present
    if 'TL' not in df.columns:
        tl_series = _calculate_trophic_levels(model)
        df = df.merge(tl_series.to_frame(), left_on='Group', right_index=True, how='left')

    df = df[df['Biomass'] > 0].sort_values('TL')

    if len(df) < 2:
        return 0.0

    # Fit linear regression: log10(biomass) vs index
    biomass = df['Biomass'].values
    x = np.arange(len(biomass))
    slope, _ = np.polyfit(x, np.log10(biomass), 1)

    return float(slope)


def calculate_biomass_range(model: RpathParams) -> float:
    """Calculate the range of biomasses (log10 scale).

    Large ranges (>6) may indicate missing groups or unrealistic values.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters

    Returns
    -------
    float
        Log10 of (max_biomass / min_biomass)
    """
    df = model.model[model.model['Type'].isin([0, 1, 2])].copy()
    biomass = df[df['Biomass'] > 0]['Biomass']

    if len(biomass) < 2:
        return 0.0

    return float(np.log10(biomass.max() / biomass.min()))


def calculate_predator_prey_ratios(model: RpathParams) -> pd.DataFrame:
    """Calculate biomass ratios between predators and their prey.

    High ratios (>1) suggest insufficient prey biomass to support predator
    consumption. Typical ratios: 0.01 to 0.5.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters

    Returns
    -------
    pd.DataFrame
        Columns: ['Predator', 'Prey_Biomass', 'Predator_Biomass', 'Ratio']
    """
    results = []

    # Get living groups (exclude detritus and fleets)
    living = model.model[model.model['Type'].isin([0, 1])].copy()

    for pred_idx, pred_row in living.iterrows():
        predator = pred_row['Group']
        pred_biomass = pred_row['Biomass']

        if pred_biomass <= 0:
            continue

        # Get diet for this predator
        if predator in model.diet.columns:
            diet = model.diet[predator]
            prey_with_diet = diet[diet > 0]

            if len(prey_with_diet) == 0:
                continue

            # Sum biomass of all prey
            prey_biomass = 0.0
            for prey_name in prey_with_diet.index:
                if prey_name in model.model['Group'].values:
                    prey_biom = model.model[model.model['Group'] == prey_name]['Biomass']
                    if not prey_biom.empty and prey_biom.iloc[0] > 0:
                        prey_biomass += prey_biom.iloc[0]

            if prey_biomass > 0:
                ratio = pred_biomass / prey_biomass
                results.append({
                    'Predator': predator,
                    'Prey_Biomass': prey_biomass,
                    'Predator_Biomass': pred_biomass,
                    'Ratio': ratio
                })

    return pd.DataFrame(results)


def calculate_vital_rate_ratios(
    model: RpathParams,
    rate_name: str = 'PB'
) -> pd.DataFrame:
    """Calculate vital rate ratios between predators and prey.

    Examines if predator rates are appropriately lower than prey rates
    (metabolic theory prediction).

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters
    rate_name : str, default 'PB'
        Rate to analyze: 'PB', 'QB', or custom column name

    Returns
    -------
    pd.DataFrame
        Columns: ['Predator', 'Prey_Rate_Mean', 'Predator_Rate', 'Ratio']
    """
    results = []

    living = model.model[model.model['Type'].isin([0, 1])].copy()

    # Check if rate column exists
    if rate_name not in living.columns:
        return pd.DataFrame(columns=['Predator', 'Prey_Rate_Mean', 'Predator_Rate', 'Ratio'])

    for pred_idx, pred_row in living.iterrows():
        predator = pred_row['Group']
        pred_rate = pred_row[rate_name]

        if pd.isna(pred_rate) or pred_rate <= 0:
            continue

        # Get prey rates
        if predator in model.diet.columns:
            diet = model.diet[predator]
            prey_with_diet = diet[diet > 0]

            if len(prey_with_diet) == 0:
                continue

            prey_rates = []
            for prey_name in prey_with_diet.index:
                if prey_name in model.model['Group'].values:
                    prey_rate_val = model.model[model.model['Group'] == prey_name][rate_name]
                    if not prey_rate_val.empty and not pd.isna(prey_rate_val.iloc[0]) and prey_rate_val.iloc[0] > 0:
                        prey_rates.append(prey_rate_val.iloc[0])

            if len(prey_rates) > 0:
                prey_mean = np.mean(prey_rates)
                ratio = pred_rate / prey_mean
                results.append({
                    'Predator': predator,
                    'Prey_Rate_Mean': prey_mean,
                    'Predator_Rate': pred_rate,
                    'Ratio': ratio
                })

    return pd.DataFrame(results)


def plot_biomass_vs_trophic_level(
    model: RpathParams,
    exclude_groups: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """Plot biomass vs trophic level with group labels.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters
    exclude_groups : list of str, optional
        Groups to exclude (e.g., homeotherms, detritus)
    figsize : tuple, default (8, 6)
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Prepare data
    df = model.model[model.model['Type'].isin([0, 1, 2])].copy()
    df = df[df['Biomass'] > 0]

    # Calculate TL if not present
    if 'TL' not in df.columns:
        tl_series = _calculate_trophic_levels(model)
        df = df.merge(tl_series.to_frame(), left_on='Group', right_index=True, how='left')

    if exclude_groups:
        df = df[~df['Group'].isin(exclude_groups)]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(df['TL'], df['Biomass'], alpha=0.6, s=50)
    ax.set_yscale('log')
    ax.set_xlabel('Trophic Level', fontsize=12)
    ax.set_ylabel('Biomass (t/km²)', fontsize=12)
    ax.set_title('Biomass vs Trophic Level', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add group labels (sample if too many)
    if len(df) <= 30:
        for _idx, row in df.iterrows():
            ax.annotate(
                row['Group'],
                (row['TL'], row['Biomass']),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.tight_layout()
    return fig


def plot_vital_rate_vs_trophic_level(
    model: RpathParams,
    rate_name: str = 'PB',
    exclude_groups: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """Plot vital rate vs trophic level.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters
    rate_name : str, default 'PB'
        Rate to plot: 'PB', 'QB', etc.
    exclude_groups : list of str, optional
        Groups to exclude
    figsize : tuple, default (8, 6)
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Prepare data
    df = model.model[model.model['Type'].isin([0, 1])].copy()

    if rate_name not in df.columns:
        raise ValueError(f"Rate '{rate_name}' not found in model")

    df = df[df[rate_name] > 0]

    # Calculate TL if not present
    if 'TL' not in df.columns:
        tl_series = _calculate_trophic_levels(model)
        df = df.merge(tl_series.to_frame(), left_on='Group', right_index=True, how='left')

    if exclude_groups:
        df = df[~df['Group'].isin(exclude_groups)]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df['TL'], df[rate_name], alpha=0.6, s=50, c='steelblue')
    ax.set_yscale('log')
    ax.set_xlabel('Trophic Level', fontsize=12)
    ax.set_ylabel(f'{rate_name} (per year)', fontsize=12)
    ax.set_title(f'{rate_name} vs Trophic Level', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add labels for interesting points
    if len(df) <= 20:
        for _idx, row in df.iterrows():
            ax.annotate(
                row['Group'],
                (row['TL'], row[rate_name]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )

    plt.tight_layout()
    return fig


def generate_prebalance_report(model: RpathParams) -> Dict:
    """Generate comprehensive pre-balance diagnostic report.

    Parameters
    ----------
    model : RpathParams
        Unbalanced Rpath parameters

    Returns
    -------
    dict
        Dictionary with diagnostic results:
        - 'biomass_slope': float
        - 'biomass_range': float
        - 'predator_prey_ratios': DataFrame
        - 'pb_ratios': DataFrame
        - 'qb_ratios': DataFrame
        - 'warnings': list of str
    """
    report = {}
    warnings = []

    # Biomass diagnostics
    report['biomass_slope'] = calculate_biomass_slope(model)
    report['biomass_range'] = calculate_biomass_range(model)

    if report['biomass_range'] > 6:
        warnings.append(f"Large biomass range ({report['biomass_range']:.1f} orders of magnitude) - check for missing groups or unrealistic values")

    if abs(report['biomass_slope']) > 2:
        warnings.append(f"Steep biomass slope ({report['biomass_slope']:.2f}) - unusual trophic structure")

    # Predator-prey ratios
    report['predator_prey_ratios'] = calculate_predator_prey_ratios(model)

    if len(report['predator_prey_ratios']) > 0:
        high_ratios = report['predator_prey_ratios'][report['predator_prey_ratios']['Ratio'] > 1.0]
        if len(high_ratios) > 0:
            for _, row in high_ratios.iterrows():
                warnings.append(f"{row['Predator']}: predator/prey ratio = {row['Ratio']:.2f} (>1, may be unsustainable)")

    # Vital rate ratios
    if 'PB' in model.model.columns:
        report['pb_ratios'] = calculate_vital_rate_ratios(model, 'PB')
    else:
        report['pb_ratios'] = pd.DataFrame()

    if 'QB' in model.model.columns:
        report['qb_ratios'] = calculate_vital_rate_ratios(model, 'QB')
    else:
        report['qb_ratios'] = pd.DataFrame()

    report['warnings'] = warnings

    return report


def print_prebalance_summary(report: Dict) -> None:
    """Print formatted pre-balance diagnostic summary.

    Parameters
    ----------
    report : dict
        Report from generate_prebalance_report()
    """
    print("=" * 60)
    print("PRE-BALANCE DIAGNOSTIC REPORT")
    print("=" * 60)
    print()

    print("BIOMASS DIAGNOSTICS:")
    print(f"  Biomass range: {report['biomass_range']:.2f} orders of magnitude")
    print(f"  Biomass slope: {report['biomass_slope']:.3f}")
    print()

    if len(report['predator_prey_ratios']) > 0:
        print("PREDATOR-PREY BIOMASS RATIOS:")
        print("  Top 5 highest ratios:")
        top5 = report['predator_prey_ratios'].nlargest(5, 'Ratio')
        for _, row in top5.iterrows():
            print(f"    {row['Predator']}: {row['Ratio']:.3f}")
        print()

    if len(report.get('pb_ratios', [])) > 0:
        print("P/B RATE RATIOS (Predator/Prey):")
        print(f"  Mean ratio: {report['pb_ratios']['Ratio'].mean():.2f}")
        print()

    if len(report['warnings']) > 0:
        print("WARNINGS:")
        for i, warning in enumerate(report['warnings'], 1):
            print(f"  {i}. {warning}")
    else:
        print("No major issues detected!")

    print()
    print("=" * 60)
