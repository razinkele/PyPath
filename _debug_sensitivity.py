import numpy as np
from pypath.core.sensitivity import _generate_morris_trajectories

k = 3
n_traj = 2
traj = _generate_morris_trajectories(k, n_traj, n_levels=4, rng=np.random.default_rng(42))
print('traj shape:', traj.shape)
print('traj:')
print(traj)
print()
for t in range(n_traj):
    start = t * (k + 1)
    print(f'trajectory {t}:')
    for step in range(k):
        diff = traj[start + step + 1] - traj[start + step]
        n_changed = np.sum(np.abs(diff) > 1e-10)
        print(f'  step {step}: diff={diff}, n_changed={n_changed}')

print('\nvalid_grid for n_levels=4:')
import numpy as np
n_levels = 4
delta = n_levels / (2 * (n_levels - 1))
grid_values = np.linspace(0, 1, n_levels)
valid_grid = grid_values[(grid_values >= delta) & (grid_values <= 1.0 - delta)]
print(f'delta={delta}, grid={grid_values}, valid_grid={valid_grid}')
