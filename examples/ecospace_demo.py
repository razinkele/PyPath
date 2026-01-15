"""
ECOSPACE Demonstration Script

This example demonstrates basic ECOSPACE functionality:
1. Creating spatial grids
2. Configuring habitat preferences
3. Setting up dispersal parameters
4. Allocating fishing effort spatially
5. Visualizing spatial patterns

Note: This is a simplified demonstration. For full ecosystem simulations,
use with a complete Ecopath/Ecosim model.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypath.spatial import (
    allocate_gravity,
    allocate_port_based,
    allocate_uniform,
    create_1d_grid,
    create_regular_grid,
    diffusion_flux,
    habitat_advection,
)


def demo_grid_creation():
    """Demonstrate creating different types of spatial grids."""
    print("=" * 60)
    print("DEMO 1: Grid Creation")
    print("=" * 60)

    # 1. Regular 2D grid
    print("\n1. Creating 5×5 regular grid...")
    grid_2d = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
    print(f"   > Created {grid_2d.n_patches} patches")
    print(f"   > {grid_2d.adjacency_matrix.nnz // 2} connections")

    # 2. 1D transect
    print("\n2. Creating 1D transect (10 patches)...")
    grid_1d = create_1d_grid(n_patches=10, spacing=1.0)
    print(f"   > Created {grid_1d.n_patches} patches")
    print(f"   > {grid_1d.adjacency_matrix.nnz // 2} connections")

    # Visualize grids
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 2D grid
    ax1.set_title("5×5 Regular Grid")
    for i in range(grid_2d.n_patches):
        c = grid_2d.patch_centroids[i]
        ax1.scatter(c[0], c[1], s=200, c="steelblue", edgecolors="black", linewidths=2)
        ax1.text(
            c[0],
            c[1],
            str(i),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Plot edges
    rows, cols = grid_2d.adjacency_matrix.nonzero()
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        if i < j:
            p1, p2 = grid_2d.patch_centroids[i], grid_2d.patch_centroids[j]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "gray", alpha=0.3, linewidth=1)

    ax1.set_xlabel("X (longitude)")
    ax1.set_ylabel("Y (latitude)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot 1D grid
    ax2.set_title("1D Transect (10 patches)")
    for i in range(grid_1d.n_patches):
        c = grid_1d.patch_centroids[i]
        ax2.scatter(c[0], c[1], s=300, c="steelblue", edgecolors="black", linewidths=2)
        ax2.text(
            c[0],
            c[1],
            str(i),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    ax2.set_xlabel("Distance from shore (km)")
    ax2.set_ylabel("")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 9.5)

    plt.tight_layout()
    plt.savefig("ecospace_demo_grids.png", dpi=150, bbox_inches="tight")
    print("\n   > Saved visualization: ecospace_demo_grids.png")


def demo_habitat_patterns():
    """Demonstrate different habitat patterns."""
    print("\n" + "=" * 60)
    print("DEMO 2: Habitat Patterns")
    print("=" * 60)

    grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
    n_patches = 25

    # Create different habitat patterns
    patterns = {}

    # 1. Uniform
    print("\n1. Uniform habitat...")
    patterns["Uniform"] = np.ones(n_patches) * 0.8

    # 2. Horizontal gradient
    print("2. Horizontal gradient (W->E)...")
    x_coords = grid.patch_centroids[:, 0]
    patterns["Horizontal Gradient"] = (x_coords - x_coords.min()) / (
        x_coords.max() - x_coords.min()
    )

    # 3. Core-periphery
    print("3. Core-periphery...")
    center = grid.patch_centroids.mean(axis=0)
    distances = np.linalg.norm(grid.patch_centroids - center, axis=1)
    patterns["Core-Periphery"] = 1 - (distances / distances.max()) ** 2

    # 4. Patchy
    print("4. Patchy (random)...")
    np.random.seed(42)
    patterns["Patchy"] = np.random.uniform(0.2, 1.0, n_patches)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, habitat) in enumerate(patterns.items()):
        ax = axes[idx]
        scatter = ax.scatter(
            grid.patch_centroids[:, 0],
            grid.patch_centroids[:, 1],
            c=habitat,
            s=400,
            cmap="YlGn",
            vmin=0,
            vmax=1,
            edgecolors="black",
            linewidths=2,
        )
        plt.colorbar(scatter, ax=ax, label="Habitat Quality")
        ax.set_title(f"{name} Habitat", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (longitude)")
        ax.set_ylabel("Y (latitude)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("ecospace_demo_habitat.png", dpi=150, bbox_inches="tight")
    print("\n   > Saved visualization: ecospace_demo_habitat.png")


def demo_dispersal_movement():
    """Demonstrate dispersal and movement."""
    print("\n" + "=" * 60)
    print("DEMO 3: Dispersal & Movement")
    print("=" * 60)

    grid = create_1d_grid(n_patches=10, spacing=1.0)

    # Initial biomass: concentrated in middle
    biomass = np.zeros(10)
    biomass[5] = 100.0

    # Habitat: better on the right
    habitat_preference = np.linspace(0.2, 1.0, 10)

    # Calculate fluxes
    print("\n1. Calculating diffusion flux...")
    diffusion = diffusion_flux(
        biomass_vector=biomass,
        dispersal_rate=5.0,
        grid=grid,
        adjacency=grid.adjacency_matrix,
    )

    print("2. Calculating habitat advection...")
    advection = habitat_advection(
        biomass_vector=biomass,
        habitat_preference=habitat_preference,
        gravity_strength=0.5,
        grid=grid,
        adjacency=grid.adjacency_matrix,
    )

    combined = diffusion + advection

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Initial biomass
    axes[0, 0].bar(range(10), biomass, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Initial Biomass", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Patch")
    axes[0, 0].set_ylabel("Biomass")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Habitat preference
    axes[0, 1].bar(
        range(10), habitat_preference, color="green", alpha=0.7, edgecolor="black"
    )
    axes[0, 1].set_title("Habitat Preference", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Patch")
    axes[0, 1].set_ylabel("Quality (0-1)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Diffusion flux
    colors_diff = ["red" if x < 0 else "blue" for x in diffusion]
    axes[1, 0].bar(
        range(10), diffusion, color=colors_diff, alpha=0.7, edgecolor="black"
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title(
        "Diffusion Flux (Random Dispersal)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Patch")
    axes[1, 0].set_ylabel("Net Flux")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Combined flux
    colors_comb = ["red" if x < 0 else "blue" for x in combined]
    axes[1, 1].bar(range(10), combined, color=colors_comb, alpha=0.7, edgecolor="black")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title(
        "Combined Flux (Diffusion + Advection)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Patch")
    axes[1, 1].set_ylabel("Net Flux")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("ecospace_demo_dispersal.png", dpi=150, bbox_inches="tight")
    print("\n   > Saved visualization: ecospace_demo_dispersal.png")

    # Check conservation
    print(f"\n   Diffusion flux sum: {diffusion.sum():.10f} (should be ~0)")
    print(f"   Advection flux sum: {advection.sum():.10f} (should be ~0)")
    print(f"   Combined flux sum: {combined.sum():.10f} (should be ~0)")


def demo_spatial_fishing():
    """Demonstrate spatial fishing effort allocation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Spatial Fishing Effort")
    print("=" * 60)

    grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
    n_patches = 25
    total_effort = 100.0

    # Demonstration biomass (high in center)
    center = grid.patch_centroids.mean(axis=0)
    distances = np.linalg.norm(grid.patch_centroids - center, axis=1)
    biomass_demo = np.zeros((2, n_patches))
    biomass_demo[1, :] = 50 * np.exp(-distances / 2)

    allocations = {}

    # 1. Uniform
    print("\n1. Uniform allocation...")
    allocations["Uniform"] = allocate_uniform(n_patches, total_effort)

    # 2. Gravity (biomass-weighted)
    print("2. Gravity allocation (alpha=1.0)...")
    allocations["Gravity (alpha=1.0)"] = allocate_gravity(
        biomass=biomass_demo,
        target_groups=[1],
        total_effort=total_effort,
        alpha=1.0,
        beta=0.0,
    )

    # 3. Gravity (alpha=2.0, stronger concentration)
    print("3. Gravity allocation (alpha=2.0)...")
    allocations["Gravity (alpha=2.0)"] = allocate_gravity(
        biomass=biomass_demo,
        target_groups=[1],
        total_effort=total_effort,
        alpha=2.0,
        beta=0.0,
    )

    # 4. Port-based
    print("4. Port-based allocation...")
    allocations["Port-based"] = allocate_port_based(
        grid=grid,
        port_patches=np.array([0, 4, 20, 24]),  # Four corners
        total_effort=total_effort,
        beta=1.5,
    )

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, effort) in enumerate(allocations.items()):
        ax = axes[idx]
        scatter = ax.scatter(
            grid.patch_centroids[:, 0],
            grid.patch_centroids[:, 1],
            c=effort,
            s=effort * 15,  # Size proportional to effort
            cmap="Reds",
            edgecolors="black",
            linewidths=2,
            vmin=0,
            vmax=effort.max(),
        )
        plt.colorbar(scatter, ax=ax, label="Fishing Effort")
        ax.set_title(f"{name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (longitude)")
        ax.set_ylabel("Y (latitude)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Add validation text
        ax.text(
            0.02,
            0.98,
            f"Total: {effort.sum():.1f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig("ecospace_demo_fishing.png", dpi=150, bbox_inches="tight")
    print("\n   > Saved visualization: ecospace_demo_fishing.png")

    # Validate conservation
    print("\nValidation:")
    for name, effort in allocations.items():
        print(f"   {name}: Total = {effort.sum():.2f} (should be 100.0)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ECOSPACE DEMONSTRATION")
    print("=" * 60)
    print("\nThis script demonstrates core ECOSPACE functionality:")
    print("  1. Grid creation (regular, 1D transect)")
    print("  2. Habitat patterns (uniform, gradient, patchy, core-periphery)")
    print("  3. Dispersal & movement (diffusion, advection)")
    print("  4. Spatial fishing (uniform, gravity, port-based)")
    print("\nGenerating visualizations...")

    # Run demos
    demo_grid_creation()
    demo_habitat_patterns()
    demo_dispersal_movement()
    demo_spatial_fishing()

    print("\n" + "=" * 60)
    print("COMPLETE: All demonstrations finished successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - ecospace_demo_grids.png")
    print("  - ecospace_demo_habitat.png")
    print("  - ecospace_demo_dispersal.png")
    print("  - ecospace_demo_fishing.png")
    print("\nNext steps:")
    print("  - View generated PNG files")
    print("  - Read docs/ECOSPACE_USER_GUIDE.md for full documentation")
    print("  - Try the Shiny app: shiny run app/app.py")
    print("  - Integrate with your Ecopath/Ecosim models")
    print()


if __name__ == "__main__":
    main()
