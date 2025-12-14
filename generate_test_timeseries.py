"""Generate artificial time series data for testing Bayesian optimization.

This script:
1. Loads a test Ecopath model
2. Runs Ecosim with known "true" parameters
3. Adds realistic noise to simulate measurement error
4. Saves observed data for optimization testing
"""

import numpy as np
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pypath.io.ewemdb import read_ewemdb
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run


def generate_artificial_timeseries(
    model_path: str,
    output_path: str,
    true_params: dict,
    groups_to_observe: list,
    years: range,
    noise_level: float = 0.1,
    random_seed: int = 42
):
    """Generate artificial time series data.

    Parameters
    ----------
    model_path : str
        Path to Ecopath model database
    output_path : str
        Path to save generated data
    true_params : dict
        "True" parameter values to use for simulation
        Example: {'vulnerability': 2.5, 'VV_3': 3.0}
    groups_to_observe : list
        List of group indices to generate observed data for
    years : range
        Years to simulate
    noise_level : float
        Standard deviation of multiplicative noise (0.1 = 10% noise)
    random_seed : int
        Random seed for reproducibility
    """
    np.random.seed(random_seed)

    print("=" * 70)
    print("GENERATING ARTIFICIAL TIME SERIES FOR OPTIMIZATION TESTING")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading model: {Path(model_path).name}")
    params = read_ewemdb(model_path, scenario=1)
    print(f"   Loaded {len(params.model)} groups")

    # Balance Ecopath
    print("\n2. Balancing Ecopath model")
    model = rpath(params)
    print(f"   Model balanced successfully")
    print(f"   Living groups: {model.NUM_LIVING}")

    # Create scenario with true parameters
    print("\n3. Creating scenario with 'true' parameters")
    scenario = rsim_scenario(model, params, years=years)

    # Update with true parameters
    for param_name, value in true_params.items():
        print(f"   Setting {param_name} = {value:.4f}")
        if param_name == 'vulnerability':
            scenario.params.VV[:] = value
        elif param_name.startswith('VV_'):
            group_idx = int(param_name.split('_')[1])
            scenario.params.VV[group_idx] = value
        elif param_name.startswith('QQ_'):
            link_idx = int(param_name.split('_')[1])
            scenario.params.QQ[link_idx] = value

    # Run simulation
    print("\n4. Running Ecosim simulation")
    result = rsim_run(scenario, method='RK4')
    print(f"   Simulation completed")
    if result.crash_year > 0:
        print(f"   Warning: Crash detected at year {result.crash_year}")

    # Extract and add noise to biomass
    print(f"\n5. Generating observed data with {noise_level*100:.1f}% noise")
    observed_data = {}

    for group_idx in groups_to_observe:
        group_name = model.Group[group_idx]

        # Get true biomass
        true_biomass = result.annual_Biomass[:, group_idx]

        # Add multiplicative log-normal noise
        # log-normal ensures biomass stays positive
        noise = np.random.lognormal(0, noise_level, size=len(true_biomass))
        noisy_biomass = true_biomass * noise

        observed_data[group_idx] = noisy_biomass

        print(f"   Group {group_idx} ({group_name}):")
        print(f"     Mean biomass: {np.mean(true_biomass):.4f}")
        print(f"     Noise std: {np.std(noise):.4f}")
        print(f"     Signal-to-noise ratio: {np.mean(true_biomass) / np.std(noisy_biomass - true_biomass):.2f}")

    # Package data
    data = {
        'model_path': model_path,
        'true_params': true_params,
        'observed_data': observed_data,
        'years': years,
        'noise_level': noise_level,
        'groups_to_observe': groups_to_observe,
        'group_names': {idx: model.Group[idx] for idx in groups_to_observe},
        'true_biomass': {idx: result.annual_Biomass[:, idx] for idx in groups_to_observe},
        'random_seed': random_seed
    }

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n6. Data saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {Path(model_path).name}")
    print(f"Simulation years: {len(years)} ({min(years)} to {max(years)})")
    print(f"True parameters: {true_params}")
    print(f"Observed groups: {len(observed_data)}")
    for idx in observed_data.keys():
        print(f"  - Group {idx}: {model.Group[idx]}")
    print(f"Noise level: {noise_level*100:.1f}%")
    print(f"Output file: {output_path}")
    print("=" * 70)

    return data


if __name__ == '__main__':
    # Configuration
    model_path = "Data/LT2022_0.5ST_final7.eweaccdb"

    # True parameter values (these will be "hidden" and optimizer will try to find them)
    true_params = {
        'vulnerability': 2.5,  # True vulnerability value
        'VV_1': 3.5,          # Herring
        'VV_3': 2.8,          # Sand-eels
    }

    # Groups to generate observed data for
    # 1 = Herring, 2 = Zooplankton, 3 = Sand-eels, 4 = Sprat
    groups_to_observe = [1, 2, 3, 4]

    # Simulation settings
    years = range(1, 31)  # 30 years
    noise_level = 0.15    # 15% noise
    random_seed = 42

    # Generate data
    data = generate_artificial_timeseries(
        model_path=model_path,
        output_path="test_timeseries_data.pkl",
        true_params=true_params,
        groups_to_observe=groups_to_observe,
        years=years,
        noise_level=noise_level,
        random_seed=random_seed
    )

    # Visualize
    print("\nGenerating visualization...")
    try:
        import matplotlib.pyplot as plt

        n_groups = len(groups_to_observe)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        years_list = list(years)
        for i, group_idx in enumerate(groups_to_observe):
            group_name = data['group_names'][group_idx]

            # Plot true vs observed
            axes[i].plot(years_list, data['true_biomass'][group_idx],
                        'b-', label='True biomass', linewidth=2)
            axes[i].plot(years_list, data['observed_data'][group_idx],
                        'ro', label='Observed (noisy)', markersize=5, alpha=0.7)
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Biomass')
            axes[i].set_title(f'{group_name} (Group {group_idx})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('test_timeseries_visualization.png', dpi=300, bbox_inches='tight')
        print("Saved visualization: test_timeseries_visualization.png")

    except ImportError:
        print("Matplotlib not available for visualization")

    print("\nTest data generation complete!")
    print("\nNext steps:")
    print("1. Run test_bayesian_optimization.py to optimize parameters")
    print("2. Optimizer will try to recover the true parameter values from noisy data")
