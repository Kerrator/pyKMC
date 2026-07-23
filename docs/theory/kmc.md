# Kinetic Monte Carlo

Kinetic Monte Carlo (KMC) evolves a system through a sequence of
state-to-state transitions, selecting each transition with the correct
relative probability and advancing a physically meaningful clock. Where
molecular dynamics is limited to nanoseconds by the vibrational time step, KMC
reaches seconds and beyond by treating only the rare transitions — the
vibrations in between are folded into the rate constants (see
[Transition State Theory](tst.md)).

## The master equation

The probability $P_i(t)$ of finding the system in state $i$ obeys

$$
\frac{dP_i(t)}{dt}
  = \sum_{j \neq i} \bigl[ k_{ji} P_j(t) - k_{ij} P_i(t) \bigr],
$$

where $k_{ij}$ is the rate constant of the transition $i \to j$. A KMC
simulation generates stochastic trajectories whose ensemble samples this
evolution exactly, provided the rate catalog is complete and the transitions
are Markovian (memoryless). The Markov assumption is inherited from
transition state theory: because the system equilibrates in a basin long
before it escapes, the escape probability per unit time is constant and the
next transition is independent of how the state was reached.

## Rejection-free selection (BKL / n-fold way)

Given the $M$ possible events from the current state with rates $k_i$, form
the total rate

$$
k_\text{tot} = \sum_{i=1}^{M} k_i ,
$$

draw a uniform random number $u_1 \in (0, 1]$, and pick the event $m$
satisfying

$$
\sum_{i=1}^{m-1} k_i \;<\; u_1\, k_\text{tot} \;\le\; \sum_{i=1}^{m} k_i .
$$

Every step executes an event — no moves are rejected — which is why this
algorithm (Bortz–Kalos–Lebowitz, also called the *n-fold way*) is the standard
for KMC.

## Time advance

After each event the clock advances by a stochastic increment drawn from the
exponential first-escape distribution:

$$
\Delta t = -\frac{\ln u_2}{k_\text{tot}}, \qquad u_2 \in (0, 1],
$$

with mean $\langle \Delta t \rangle = 1 / k_\text{tot}$. Time steps are large
when barriers are high (slow dynamics) and small when fast events dominate.

This is exactly what pyKMC's selection does (`pykmc.algorithms.rejection_free`):
one uniform draw picks the event from the cumulative rate sum, a second draws
$\Delta t$. (The implementation draws both numbers with Python's
`random.random()`, whose range is $[0, 1)$ rather than the $(0, 1]$ written
above; the zero draw that would break the logarithm has probability zero and
is not explicitly guarded.) Rates are handled in $\text{ps}^{-1}$ (see the
units note in [Transition State Theory](tst.md)), so $\Delta t$ is in
picoseconds; the accumulated simulation time is reported in seconds.

## Lattice vs on-the-fly KMC

- **Lattice KMC** maps states onto a fixed lattice with a precomputed event
  catalog. Fast, but blind to events outside the catalog and to off-lattice
  relaxation.
- **On-the-fly (adaptive) KMC** — pyKMC's approach — discovers events from the
  actual atomic configuration as the simulation runs, using saddle-point
  searches. No lattice and no prior knowledge of the transitions is assumed,
  at the cost of the searches themselves; reuse across equivalent
  environments amortizes that cost.

What makes the on-the-fly variant possible is that each KMC step needs only
the rates *out of the current state* — the global catalog never has to be
enumerated in advance, only the escape paths of the configuration at hand.

## pyKMC's loop

Each pyKMC step classifies atomic environments, searches for events at new
environments (pARTn), uses IRA to map known event geometries onto recurring
equivalent environments and refines their saddles with pARTn, selects with
the BKL algorithm above, reconstructs the selected transition, advances
$\Delta t$, and applies the chosen event — see the
[Algorithm Overview](general_algorithm.md).

## Limitation: low-barrier trapping and basins

When a group of states is connected by barriers $\ll$ the barriers leading
out, KMC burns its steps flickering inside the group while the clock barely
advances. pyKMC's [basin acceleration](../user_guide/basins.md) detects such
groups, solves the absorbing Markov chain for the exit time and exit state,
and replaces the flicker with a single super-event.

## References

- A. B. Bortz, M. H. Kalos, and J. L. Lebowitz, *J. Comput. Phys.* **17**, 10
  (1975) — the rejection-free (n-fold way) algorithm.
- D. T. Gillespie, *J. Comput. Phys.* **22**, 403 (1976) — stochastic
  simulation algorithm.
- A. F. Voter, *Introduction to the Kinetic Monte Carlo Method*, in Radiation
  Effects in Solids (Springer, 2007) — the standard introduction.
- G. Henkelman and H. Jónsson, *J. Chem. Phys.* **115**, 9657 (2001) —
  long-time scale KMC with on-the-fly saddle searches.
