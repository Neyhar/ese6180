# Context versus required execution length

Fixed tracking RMS tolerance: **0.05** (absolute state units).
A cell passes only when every evaluated seed has finite tracking RMS at or below the tolerance and boundary spectral radius strictly below one.
This is descriptive analysis of the supplied evaluation seeds, not a statistical guarantee. The minimum is over tested execution lengths only. No feasible setting is recorded as None, not as zero.

| Observation | Data | Context | Shortest execution | Worst tracking RMS |
|---|---|---:|---:|---:|
| full | initial | 1 | 1 | 6.192e-09 |
| full | initial | 2 | None | — |
| full | initial | 3 | None | — |
| full | initial | 4 | None | — |
| full | initial | 6 | None | — |
| full | initial | 8 | None | — |
| full | narrow | 1 | None | — |
| full | narrow | 2 | None | — |
| full | narrow | 3 | None | — |
| full | narrow | 4 | None | — |
| full | narrow | 6 | None | — |
| full | narrow | 8 | None | — |
| full | persistent | 1 | 1 | 3.322e-09 |
| full | persistent | 2 | 1 | 3.382e-09 |
| full | persistent | 3 | 1 | 3.472e-09 |
| full | persistent | 4 | 1 | 3.558e-09 |
| full | persistent | 6 | 1 | 3.697e-09 |
| full | persistent | 8 | 1 | 3.947e-09 |
| partial | initial | 1 | None | — |
| partial | initial | 2 | None | — |
| partial | initial | 3 | None | — |
| partial | initial | 4 | None | — |
| partial | initial | 6 | None | — |
| partial | initial | 8 | None | — |
| partial | narrow | 1 | None | — |
| partial | narrow | 2 | None | — |
| partial | narrow | 3 | None | — |
| partial | narrow | 4 | None | — |
| partial | narrow | 6 | None | — |
| partial | narrow | 8 | None | — |
| partial | persistent | 1 | None | — |
| partial | persistent | 2 | None | — |
| partial | persistent | 3 | None | — |
| partial | persistent | 4 | None | — |
| partial | persistent | 6 | None | — |
| partial | persistent | 8 | None | — |

A decreasing feasible execution length with increasing context supports the proposed relationship in that setting. Constant length one indicates no further reduction is possible. None does not prove infeasibility outside the tested grid.
Choose the tolerance before a confirmatory run; exploring thresholds on existing results is exploratory. Confirm any selected setting with fresh training/evaluation seeds before making a generalization claim.
