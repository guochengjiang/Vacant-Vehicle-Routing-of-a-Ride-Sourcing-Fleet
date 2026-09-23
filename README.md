# Fleet Routing

Link-based fleet routing with a shared optimization and simulation implementation
for Shanghai and a synthetic small network. All comments and documentation are in English.

## Project layout

| Location | Responsibility |
| --- | --- |
| `configs/shanghai.py` | User-editable Shanghai settings and selected input/output files |
| `configs/small_network.py` | Original small-network settings and solver order |
| `scripts/` | Runnable entry points |
| `src/fleet_routing/model/` | Policy, features and fixed-point equations |
| `src/fleet_routing/optimization/` | Optimizer, sensitivities and checkpoints |
| `src/fleet_routing/networks/` | Shanghai loader and synthetic grid builder |
| `src/fleet_routing/simulation.py` | Vehicle simulation and repeated evaluation |
| `src/fleet_routing/estimation.py` | Alpha estimation from saved observations |
| `src/fleet_routing/workflows.py` | Connect loading, estimation, optimization and evaluation |
| `src/fleet_routing/config.py` | Configuration dataclass and project paths |
| `src/fleet_routing/experimental/` | Optional notebook evaluation routines |
| `tests/` | Numerical regression, workflow, topology and gradient checks |
| `data/shanghai/` | Original Shanghai input files |
| `results/` | Generated outputs; small network has a separate subfolder |
| `notebooks/` | Optional plots and exploration |
| `archive/notebook_cells.txt` | Notebook reference used for regression checks |

## Run individual stages

Run commands from the project root, or right-click the corresponding script in PyCharm.

| Script | Action |
| --- | --- |
| `python scripts/run_generate_trajectories.py` | Generate calibration data with three baseline policies |
| `python scripts/run_estimation.py` | Read saved trajectories and estimate alpha; does not simulate |
| `python scripts/run_optimization.py` | Optimize theta and save the resulting policy |
| `python scripts/run_simulation.py` | Evaluate the saved policy or selected baseline |
| `python scripts/run_manual_simulation.py` | Build pi from manually entered theta, then evaluate |
| `python scripts/run_small_network.py` | Original small-network BFGS, gradient check and Nelder-Mead; no simulation |
| `python scripts/run_tests.py` | Run automated checks without Shanghai inputs |

If trajectories already exist, skip generation. If alpha is known, enter it in
CONFIG and skip estimation. If an optimization result already exists, select it
with OPTIMIZATION_FILE and run simulation directly.

## Settings

Edit `configs/shanghai.py`, for example:

```python
CONFIG = ExperimentConfig(
    M=5000,
    demand_reference_fleet=5000,
    heuristic=None,
    alpha=None,
    n_runs=10,
    sim_hours=12,
    warmup_hours=2,
)
```

- `M` changes fleet size; it does not change demand. Demand is the original input
  rate multiplied by `demand_reference_fleet / demand_base_fleet` (default base 12000).
- `heuristic='CF'` enables the inherited competition-free fixed-point mode;
  `None` uses the normal model. The inherited CF gradient has not been re-derived
  for fixed matching probabilities. This limitation remains explicit; CF optimization
  is not validated by the normal-model gradient test.
- `n_runs=10` means ten evaluation simulations. Calibration generation instead
  runs each of its three baseline policies once.
- `sim_hours` includes `warmup_hours`. Defaults give ten hours of recorded statistics.
- `alpha=None` loads alpha from CALIBRATION_FILE for optimization. The simulator
  uses explicit passenger queues and does not consume alpha directly.
- Input paths and default output paths are project-relative, not working-directory-relative.

## Shanghai input files

Copy these files unchanged from your current `Input/` folder:

```text
c_final.npy
v_final.npy
Q_final.npy
lambda_vec_final.npy
tau_final.npy
omega_final.npy
feature_profit_final.npy
init_dist_final.npy
outgoing_links_final.pkl
```

The original Shanghai data is not included in this distribution. The loader validates
array shapes and probability rows. Time arrays c, tau and omega are interpreted in hours.

## Reuse previous results and trajectories

Copy the desired trusted pickle files from the old results folder into this project's
results folder, then set TRAJECTORY_FILE, CALIBRATION_FILE or OPTIMIZATION_FILE in
`configs/shanghai.py`. Generation never runs automatically during estimation.
Trajectory files need `trajectories` and `z_avg_combined`. Newer files also contain
demand and travel-time metadata. Legacy files without metadata require you to check
that the configured demand matches the original experiment.

Optimization files created by the modular version contain `policy`, `Parameters`
and `lambda_sim` and can be used directly. Older notebook results with only theta
should use the manual simulation entry point instead. Load only trusted pickle files.

## Manual theta to pi to simulation

In `configs/shanghai.py`, replace MANUAL_THETA with your `result.x` values, such as
`MANUAL_THETA = [7.79905344]`, and choose a descriptive MANUAL_NAME. The default `[0.0]`
is just a runnable uniform-logit example, not an estimated optimum.

Theta must correspond to the same feature order, link order and normalization used
in optimization. By default the current input features are min-max normalized with
the inherited epsilon 1e-8. To reuse an optimization file's feature matrix, set
FEATURE_REFERENCE_FILE to that file. Matching dimensions alone cannot verify link
identity: the network and feature ordering must also match. Raw-feature theta from
the old small-network experiment is not interchangeable with normalized-feature theta.

## Saved summaries

Every repeated simulation saves a pickle containing `summary`, `parameters`, and
`all_trajectories`, plus a readable JSON summary with the same filename stem.
The summary includes means, population standard deviations (`ddof=0`), reward per
run and mean vehicle distributions. Set `save_trajectories=False` to omit trajectories
from saved repeated results; individual simulations still construct them internally.
The manual entry point also saves theta, feature matrix and pi. Reusing the same
experiment filename overwrites that output; use distinct MANUAL_NAME values or copy
results before rerunning.

## Small-network experiment

`python scripts/run_small_network.py` reproduces the uploaded old experiment:
analytic BFGS, a gradient check at theta zero, then Nelder-Mead. It does not run
simulation. Both solvers start at zero. Results are saved separately and each
solver's own theta, reward, success flag and termination message are printed.

The original generator is in `networks/legacy_small.py`: 24 nodes, 76 links,
seeds 2025/2025, original connectivity and time conventions. All executable
network-generation operations are preserved. Features are raw lambda_vec,
without normalization. M=1000, alpha=0.8, beta=24, fixed-point tolerances 1e-10,
maximum 200000 inner iterations and restart interval 10000 match the old script.
Legacy topology/unit conventions are preserved for reproducibility, not endorsed
as a corrected physical network. The alternate corrected grid in `networks/small.py`
is only used by the separate grid tests; this entry point does not call it.

The uploaded script also ran numerical BFGS between analytic BFGS and Nelder-Mead.
Set RUN_NUMERICAL_BFGS=True in `configs/small_network.py` to include that extra run.
Default False follows the requested BFGS -> Nelder-Mead sequence.

The old script mistakenly printed the analytic BFGS result.x after the other
solvers. This is corrected: the final Nelder-Mead theta is approximately 0.04448242,
while analytic BFGS gives approximately 0.04449017. Both yield R=173.262769 in the
reference run. The old screenshot's final theta was therefore the BFGS theta.

## Validation and scope

See VALIDATION.md for checks performed and limitations. This refactor preserves the
available Shanghai algorithm implementation; it is not a memory or speed optimization.
The sensitivity solver still uses dense matrices. Checkpoints are warm-start snapshots,
not complete resumable BFGS state.

This package was assembled from the available modular project and old small-network
source. Any subsequent edits that exist only on your Mac are not included. Keep your
current project as a backup and compare those local edits before replacing it.
