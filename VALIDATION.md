# Validation

Validated on 2026-09-23 using the available modular source and notebook reference.

## Small-network restoration (version 4)

The default small-network entry point now reproduces the original experiment,
not the alternate corrected grid. The legacy generator has identical executable
AST to the uploaded source (only English documentation/comments changed).
Original and restored scripts were both executed locally:

| Solver | Theta | Reward | Termination |
| --- | --- | --- | --- |
| Analytic BFGS | 0.04449017 | 173.262769 | Success |
| Nelder-Mead | 0.04448242 | 173.262769 | Success; 23 iterations, 49 evaluations |

The original script's final theta print reused the BFGS result. The restored
script prints each solver's own result. An automated regression checks successful
termination, both parameter values and both rewards. The default run contains no
simulation. Numerical BFGS from the uploaded script is optional.

The specific BFGS error reported on the user's Mac was not present in the supplied
screenshots and could not be diagnosed from them. Successful local reproduction
does not establish the cause of that earlier error.

## Passed checks

- Python compilation of project modules and scripts.
- Exact policy equivalence to the archived notebook on a three-link network.
- Fixed-point state, objective and analytic-gradient parity with the notebook.
- Seeded simulation reward, transitions and statistics match the notebook.
- Alpha estimation matches the notebook on identical observations.
- Calibration generation -> saved-data estimation -> optimization -> repeated simulation.
- Estimation is tested with simulation mocked to raise if called.
- Legacy trajectory input support and rejection of mismatched demand metadata.
- JSON summary round-trip agrees with the saved numerical summary.
- The default small grid has 24 nodes and 76 directed links.
- Every legal successor starts at the current link's head.
- Grid time units, Q row sums, policy normalization and action support.
- Independent central finite-difference check of the normal-model gradient on a
  2 by 2 grid, using fresh optimizer instances (rtol 0.002, atol 0.0002).
- A 76-link smoke run completed two optimization iterations and two short
  simulations, producing finite objectives and summaries. The optimizer iteration
  cap is deliberate and does not establish convergence.
- The test entry point works when launched from a different working directory.

Run `python scripts/run_tests.py` for the repeatable regression suite. The larger
76-link smoke run was performed separately with reduced iteration/time settings.

## Boundaries

Full Shanghai runs were not performed because its original nine input files were
not supplied. No claim is made about full-network convergence, performance, memory
improvement or equivalence to unuploaded local edits. The normal-model gradient
check does not validate CF sensitivities. The inherited CF sensitivity limitation
and experimental evaluation routines remain documented in the README.

The new grid is intentionally different from the legacy small-network generator:
its successor direction and time units are corrected, and synthetic parameter
construction is explicit. It is not the standard Sioux Falls data set.
