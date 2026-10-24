\# PyPath Performance Optimisation Backlog



\## NumPy, sparse trophic links, Numba acceleration, and ensemble parallelism



\*\*Status:\*\* Proposed implementation backlog  

\*\*Scope:\*\* PyPath Ecopath/Ecosim computational core and repeated-simulation workflows  

\*\*Primary objective:\*\* Improve runtime and memory efficiency without changing validated ecological or numerical behaviour



\---



\# 1. Purpose



This document is the version-controlled source of truth for the first PyPath performance optimisation programme.



The programme concentrates on four complementary development pathways:



1\. NumPy and memory-layout optimisation;

2\. sparse active trophic-link representation;

3\. Numba compilation of numerical kernels;

4\. process-level parallel execution of independent simulations.



These approaches operate at different levels and should not be treated as competing alternatives.



The intended execution architecture is:



```text

PyPath model / EwE database

&#x20;         |

&#x20;         v

scenario validation

&#x20;         |

&#x20;         v

compile scenario

&#x20;         |

&#x20;         v

Immutable SimulationPlan

&#x20;         +

Mutable SimulationState

&#x20;         +

Reusable Workspace

&#x20;         |

&#x20;         v

Numerical kernel

&#x20;    /          \\

&#x20;NumPy          Numba

reference      optional

&#x20;    \\          /

&#x20;         v

&#x20;     Integrator

&#x20;         |

&#x20;         v

&#x20;  SimulationResult

```



Independent simulations are handled separately:



```text

Parameter/scenario ensemble

&#x20;          |

&#x20;          v

&#x20;   EnsembleExecutor

&#x20;          |

&#x20;   +------+------+------+

&#x20;   |      |      |      |

&#x20;   v      v      v      v

&#x20; run 1  run 2  run 3  run N

&#x20;   |      |      |      |

&#x20;   +------+------+------+

&#x20;          |

&#x20;          v

&#x20;    EnsembleResult

```



The single-simulation numerical engine and ensemble-execution layer must remain architecturally separate.



\---



\# 2. Development principles



\## 2.1 Correctness before speed



Performance changes must not silently change validated ecological behaviour.



Every optimised pathway must be evaluated against scientific guardrails that include, where applicable:



\- trophic flows;

\- derivative components;

\- complete derivatives;

\- one-step integration;

\- short trajectories;

\- long trajectories;

\- conservation diagnostics.



\## 2.2 One canonical numerical representation



NumPy and Numba should consume the same canonical arrays.



The architecture should avoid independent data models for:



\- NumPy;

\- Numba;

\- Cython;

\- C++;

\- Fortran.



\## 2.3 Optimise measured bottlenecks



No optimisation technology should be adopted because it is assumed to be fast.



Performance decisions must be based on representative PyPath benchmarks.



\## 2.4 Preserve the readable reference implementation



The NumPy implementation should remain scientifically inspectable even if an accelerated backend becomes available.



\## 2.5 Do not prematurely add native backends



Cython, C++, and modern Fortran remain possible future pathways.



They should only be evaluated after the NumPy, link-topology, Numba, and ensemble programme identifies a remaining material need.



\---



\# 3. Milestones



| Milestone | Purpose |

|---|---|

| M0 | Performance baseline and scientific guardrails |

| M1 | Execution architecture |

| M2 | Sparse trophic

