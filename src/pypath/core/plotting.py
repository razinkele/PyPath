"""
Plotting module for PyPath.

This module provides visualization functions for Ecopath models
and Ecosim simulation results using matplotlib and optionally plotly.

Functions include:
- Food web network diagrams
- Biomass time series
- Catch time series
- Trophic level distributions
- Mixed Trophic Impacts heatmaps

Based on Rpath's plotting functions.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np

# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Try to import networkx for food web graphs
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from pypath.core.ecopath import Rpath
from pypath.core.ecosim import RsimScenario, RsimOutput


# =============================================================================
# FOOD WEB PLOTTING
# =============================================================================

def plot_foodweb(
    rpath: Rpath,
    title: str = "Food Web",
    layout: str = 'trophic',
    node_size_by: str = 'biomass',
    edge_width_by: str = 'flow',
    show_labels: bool = True,
    min_flow: float = 0.01,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot food web network diagram.
    
    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model
    title : str
        Plot title
    layout : str
        Node layout: 'trophic' (y=TL), 'spring', 'circular'
    node_size_by : str
        What to scale node size by: 'biomass', 'production', 'equal'
    edge_width_by : str
        What to scale edge width by: 'flow', 'diet', 'equal'
    show_labels : bool
        Show group labels
    min_flow : float
        Minimum flow to show (relative to max)
    figsize : tuple
        Figure size
    cmap : str
        Colormap for trophic levels
    ax : Axes, optional
        Matplotlib axes to plot on
    
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for food web plots. Install with: pip install networkx")
    
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_total = n_living + n_dead
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(1, n_total + 1):
        G.add_node(i, 
                   tl=rpath.TL[i],
                   biomass=rpath.Biomass[i],
                   is_detritus=i > n_living)
    
    # Add edges (from prey to predator)
    max_flow = 0
    for pred in range(1, n_living + 1):
        for prey in range(1, n_total + 1):
            if rpath.DC[prey, pred] > 0:
                flow = rpath.DC[prey, pred] * rpath.QB[pred] * rpath.Biomass[pred]
                max_flow = max(max_flow, flow)
                G.add_edge(prey, pred, flow=flow, diet=rpath.DC[prey, pred])
    
    # Filter small flows
    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        if data['flow'] < min_flow * max_flow:
            edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)
    
    # Calculate layout
    if layout == 'trophic':
        # Position by trophic level (y) and spread horizontally
        pos = {}
        tl_groups = {}
        for node in G.nodes():
            tl = round(G.nodes[node]['tl'], 1)
            if tl not in tl_groups:
                tl_groups[tl] = []
            tl_groups[tl].append(node)
        
        for tl, nodes in tl_groups.items():
            n = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - (n - 1) / 2) * 0.8
                pos[node] = (x, tl)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate node sizes
    if node_size_by == 'biomass':
        max_bio = max(rpath.Biomass[1:n_total + 1])
        node_sizes = [500 + 2000 * (rpath.Biomass[i] / max_bio) for i in G.nodes()]
    elif node_size_by == 'production':
        prods = [rpath.PB[i] * rpath.Biomass[i] for i in G.nodes()]
        max_prod = max(prods) if max(prods) > 0 else 1
        node_sizes = [500 + 2000 * (p / max_prod) for p in prods]
    else:
        node_sizes = [800] * len(G.nodes())
    
    # Calculate edge widths
    if edge_width_by == 'flow':
        edge_widths = []
        for u, v in G.edges():
            w = G.edges[u, v]['flow'] / max_flow if max_flow > 0 else 0
            edge_widths.append(0.5 + 4 * w)
    elif edge_width_by == 'diet':
        edge_widths = [0.5 + 4 * G.edges[u, v]['diet'] for u, v in G.edges()]
    else:
        edge_widths = [1.5] * len(G.edges())
    
    # Node colors by trophic level
    trophic_levels = [G.nodes[i]['tl'] for i in G.nodes()]
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=trophic_levels,
        cmap=plt.cm.get_cmap(cmap),
        ax=ax,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    if show_labels:
        labels = {i: f'G{i}' for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Add colorbar for trophic levels
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap(cmap),
        norm=plt.Normalize(vmin=min(trophic_levels), vmax=max(trophic_levels))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Trophic Level')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# ECOSIM TIME SERIES PLOTS
# =============================================================================

def plot_biomass(
    output: RsimOutput,
    groups: Optional[List[int]] = None,
    relative: bool = False,
    title: str = "Biomass Time Series",
    figsize: Tuple[int, int] = (12, 6),
    legend_loc: str = 'best',
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot biomass time series from Ecosim simulation.
    
    Parameters
    ----------
    output : RsimOutput
        Simulation results
    groups : list of int, optional
        Group indices to plot (default: all living)
    relative : bool
        If True, plot relative to initial biomass
    title : str
        Plot title
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    ax : Axes, optional
        Matplotlib axes
    
    Returns
    -------
    matplotlib.Figure
    """
    biomass = output.out_Biomass_annual
    n_years, n_groups = biomass.shape
    
    if groups is None:
        # Plot all groups with significant biomass
        groups = [i for i in range(1, n_groups) if biomass[0, i] > 0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    years = np.arange(1, n_years + 1)
    
    for grp in groups:
        y = biomass[:, grp]
        if relative and y[0] > 0:
            y = y / y[0]
        ax.plot(years, y, label=f'Group {grp}', linewidth=1.5)
    
    ax.set_xlabel('Year', fontsize=11)
    ylabel = 'Relative Biomass (B/B₀)' if relative else 'Biomass'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    
    if relative:
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    ax.legend(loc=legend_loc, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_catch(
    output: RsimOutput,
    groups: Optional[List[int]] = None,
    title: str = "Catch Time Series",
    figsize: Tuple[int, int] = (12, 6),
    stacked: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot catch time series from Ecosim simulation.
    
    Parameters
    ----------
    output : RsimOutput
        Simulation results
    groups : list of int, optional
        Group indices to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    stacked : bool
        If True, create stacked area plot
    ax : Axes, optional
        Matplotlib axes
    
    Returns
    -------
    matplotlib.Figure
    """
    catch = output.out_Catch_annual
    n_years, n_groups = catch.shape
    
    if groups is None:
        # Plot groups with any catch
        groups = [i for i in range(1, n_groups) if np.sum(catch[:, i]) > 0]
    
    if not groups:
        # No catch - return empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No catch data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    years = np.arange(1, n_years + 1)
    
    if stacked:
        catch_data = [catch[:, grp] for grp in groups]
        labels = [f'Group {grp}' for grp in groups]
        ax.stackplot(years, catch_data, labels=labels, alpha=0.7)
    else:
        for grp in groups:
            ax.plot(years, catch[:, grp], label=f'Group {grp}', linewidth=1.5)
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Catch', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_biomass_grid(
    output: RsimOutput,
    groups: Optional[List[int]] = None,
    n_cols: int = 4,
    relative: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Plot biomass as a grid of subplots.
    
    Parameters
    ----------
    output : RsimOutput
        Simulation results
    groups : list of int, optional
        Group indices to plot
    n_cols : int
        Number of columns in grid
    relative : bool
        Plot relative to initial biomass
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    matplotlib.Figure
    """
    biomass = output.out_Biomass_annual
    n_years, n_groups = biomass.shape
    
    if groups is None:
        groups = [i for i in range(1, n_groups) if biomass[0, i] > 0]
    
    n_plots = len(groups)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (3 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    years = np.arange(1, n_years + 1)
    
    for idx, grp in enumerate(groups):
        ax = axes[idx]
        y = biomass[:, grp]
        
        if relative and y[0] > 0:
            y = y / y[0]
            ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        ax.plot(years, y, color='steelblue', linewidth=1.5)
        ax.set_title(f'Group {grp}', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Biomass Time Series' + (' (Relative)' if relative else ''), fontsize=12)
    plt.tight_layout()
    return fig


# =============================================================================
# ECOPATH PLOTS
# =============================================================================

def plot_trophic_spectrum(
    rpath: Rpath,
    by: str = 'biomass',
    n_bins: int = 10,
    title: str = "Trophic Spectrum",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot trophic spectrum (biomass or production by trophic level).
    
    Parameters
    ----------
    rpath : Rpath
        Balanced model
    by : str
        What to aggregate: 'biomass', 'production', 'consumption'
    n_bins : int
        Number of trophic level bins
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : Axes, optional
        Matplotlib axes
    
    Returns
    -------
    matplotlib.Figure
    """
    n_living = rpath.NUM_LIVING
    
    # Get values and trophic levels
    tl = rpath.TL[1:n_living + 1]
    
    if by == 'biomass':
        values = rpath.Biomass[1:n_living + 1]
        ylabel = 'Biomass'
    elif by == 'production':
        values = rpath.PB[1:n_living + 1] * rpath.Biomass[1:n_living + 1]
        ylabel = 'Production'
    elif by == 'consumption':
        values = rpath.QB[1:n_living + 1] * rpath.Biomass[1:n_living + 1]
        ylabel = 'Consumption'
    else:
        raise ValueError(f"Unknown 'by' value: {by}")
    
    # Create bins
    tl_min, tl_max = np.floor(np.min(tl)), np.ceil(np.max(tl))
    bins = np.linspace(tl_min, tl_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Aggregate
    aggregated = np.zeros(n_bins)
    for i in range(len(tl)):
        bin_idx = np.digitize(tl[i], bins) - 1
        bin_idx = min(bin_idx, n_bins - 1)
        aggregated[bin_idx] += values[i]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.bar(bin_centers, aggregated, width=bins[1] - bins[0], 
           color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Trophic Level', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_mti_heatmap(
    mti: np.ndarray,
    group_names: Optional[List[str]] = None,
    title: str = "Mixed Trophic Impacts",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot Mixed Trophic Impacts as a heatmap.
    
    Parameters
    ----------
    mti : np.ndarray
        MTI matrix from mixed_trophic_impacts()
    group_names : list of str, optional
        Names for groups
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    ax : Axes, optional
        Matplotlib axes
    
    Returns
    -------
    matplotlib.Figure
    """
    n = mti.shape[0]
    
    if group_names is None:
        group_names = [f'G{i}' for i in range(1, n + 1)]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Symmetric colormap around zero
    vmax = np.max(np.abs(mti))
    vmin = -vmax
    
    im = ax.imshow(mti, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Impact')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(group_names, fontsize=8)
    
    ax.set_xlabel('Impacted', fontsize=11)
    ax.set_ylabel('Impacting', fontsize=11)
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# PLOTLY INTERACTIVE PLOTS (if available)
# =============================================================================

def plot_biomass_interactive(
    output: RsimOutput,
    groups: Optional[List[int]] = None,
    relative: bool = False,
    title: str = "Biomass Time Series",
) -> Any:
    """Create interactive biomass plot with Plotly.
    
    Parameters
    ----------
    output : RsimOutput
        Simulation results
    groups : list of int, optional
        Groups to plot
    relative : bool
        Plot relative to initial
    title : str
        Plot title
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly")
    
    biomass = output.out_Biomass_annual
    n_years, n_groups = biomass.shape
    
    if groups is None:
        groups = [i for i in range(1, n_groups) if biomass[0, i] > 0]
    
    fig = go.Figure()
    
    years = np.arange(1, n_years + 1)
    
    for grp in groups:
        y = biomass[:, grp]
        if relative and y[0] > 0:
            y = y / y[0]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=y,
            mode='lines',
            name=f'Group {grp}',
            hovertemplate='Year: %{x}<br>Biomass: %{y:.4f}<extra></extra>'
        ))
    
    ylabel = 'Relative Biomass (B/B₀)' if relative else 'Biomass'
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white'
    )
    
    if relative:
        fig.add_hline(y=1, line_dash='dash', line_color='gray', opacity=0.5)
    
    return fig


def plot_foodweb_interactive(
    rpath: Rpath,
    title: str = "Food Web",
    min_flow: float = 0.01,
) -> Any:
    """Create interactive food web plot with Plotly.
    
    Parameters
    ----------
    rpath : Rpath
        Balanced model
    title : str
        Plot title
    min_flow : float
        Minimum flow to show
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly")
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for food web plots. Install with: pip install networkx")
    
    n_living = rpath.NUM_LIVING
    n_dead = rpath.NUM_DEAD
    n_total = n_living + n_dead
    
    # Build graph for layout
    G = nx.DiGraph()
    for i in range(1, n_total + 1):
        G.add_node(i, tl=rpath.TL[i], biomass=rpath.Biomass[i])
    
    max_flow = 0
    edges = []
    for pred in range(1, n_living + 1):
        for prey in range(1, n_total + 1):
            if rpath.DC[prey, pred] > 0:
                flow = rpath.DC[prey, pred] * rpath.QB[pred] * rpath.Biomass[pred]
                max_flow = max(max_flow, flow)
                G.add_edge(prey, pred)
                edges.append((prey, pred, flow))
    
    # Layout
    pos = {}
    tl_groups = {}
    for node in G.nodes():
        tl = round(rpath.TL[node], 1)
        if tl not in tl_groups:
            tl_groups[tl] = []
        tl_groups[tl].append(node)
    
    for tl, nodes in tl_groups.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * 0.8
            pos[node] = (x, tl)
    
    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f'Group {n}<br>TL: {rpath.TL[n]:.2f}<br>B: {rpath.Biomass[n]:.4f}' 
                 for n in G.nodes()]
    node_size = [10 + 30 * rpath.Biomass[n] / max(rpath.Biomass[1:n_total + 1]) 
                 for n in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f'G{n}' for n in G.nodes()],
        hovertext=node_text,
        textposition='top center',
        marker=dict(
            size=node_size,
            color=[rpath.TL[n] for n in G.nodes()],
            colorscale='Viridis',
            colorbar=dict(title='Trophic Level'),
            line_width=2
        )
    )
    
    # Edge traces
    edge_traces = []
    for prey, pred, flow in edges:
        if flow >= min_flow * max_flow:
            x0, y0 = pos[prey]
            x1, y1 = pos[pred]
            width = 1 + 4 * flow / max_flow
            
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
    
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='Trophic Level'),
        template='plotly_white'
    )
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def plot_ecosim_summary(
    output: RsimOutput,
    groups: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create summary plot with biomass, relative biomass, and catch.
    
    Parameters
    ----------
    output : RsimOutput
        Simulation results
    groups : list of int, optional
        Groups to plot
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    plot_biomass(output, groups=groups, ax=axes[0, 0], relative=False)
    axes[0, 0].set_title('Absolute Biomass')
    
    plot_biomass(output, groups=groups, ax=axes[0, 1], relative=True)
    axes[0, 1].set_title('Relative Biomass (B/B₀)')
    
    plot_catch(output, groups=groups, ax=axes[1, 0], stacked=False)
    axes[1, 0].set_title('Catch by Group')
    
    plot_catch(output, groups=groups, ax=axes[1, 1], stacked=True)
    axes[1, 1].set_title('Total Catch (Stacked)')
    
    plt.tight_layout()
    return fig


def save_plots(
    figures: Union[plt.Figure, List[plt.Figure]],
    filename: str,
    dpi: int = 150,
    format: str = 'png'
) -> None:
    """Save matplotlib figure(s) to file.
    
    Parameters
    ----------
    figures : Figure or list of Figure
        Figure(s) to save
    filename : str
        Output filename (without extension for multiple figures)
    dpi : int
        Resolution
    format : str
        Output format ('png', 'pdf', 'svg')
    """
    if isinstance(figures, plt.Figure):
        figures = [figures]
    
    if len(figures) == 1:
        figures[0].savefig(f"{filename}.{format}", dpi=dpi, bbox_inches='tight')
    else:
        for i, fig in enumerate(figures):
            fig.savefig(f"{filename}_{i+1}.{format}", dpi=dpi, bbox_inches='tight')
