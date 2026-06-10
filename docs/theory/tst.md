# Transition State Theory

Transition State Theory (TST) predicts the rate of thermally activated rare
events — transitions that cross an energy barrier much larger than the thermal
energy $k_B T$. It is the physical basis for the rate constants pyKMC assigns
to the events it discovers.

*This page is an outline introduction; each section is a starting point to be
expanded.*

## Potential energy surface and saddle points

The configuration of an $N$-atom system is a point on a $3N$-dimensional
**potential energy surface** (PES). Local minima of the PES are metastable
states; two adjacent minima are connected by a **minimum energy path** passing
through a **first-order saddle point** (a maximum along the path, a minimum in
every other direction). The **activation barrier** of the forward transition is

$$
\Delta E = E_\text{saddle} - E_\text{min}.
$$

At ordinary temperatures the system vibrates around a minimum for a long time
before a rare fluctuation carries it over a saddle — this separation of time
scales is what makes KMC possible (see [Kinetic Monte Carlo](kmc.md)).

## TST assumptions

TST computes the rate as the thermally averaged flux through a dividing
surface placed at the saddle, under these assumptions:

- **Quasi-equilibrium**: the system is in thermal equilibrium within the
  initial basin.
- **No recrossing**: a trajectory crossing the dividing surface does not
  immediately return.
- **Classical nuclei**: motion on the PES follows classical mechanics
  (no tunnelling).

## Harmonic TST and the Vineyard prefactor

Expanding the PES harmonically around the minimum and the saddle gives the
**harmonic TST** (hTST) rate with a temperature-independent prefactor
(Vineyard, 1957):

$$
k^\text{hTST}
  = \frac{\prod_{i=1}^{3N} \nu_i^{\min}}
         {\prod_{i=1}^{3N-1} \nu_i^{\ddagger}}
    \, \exp\!\left(-\frac{\Delta E}{k_B T}\right),
$$

where $\nu_i^{\min}$ are the normal-mode frequencies at the minimum and
$\nu_i^{\ddagger}$ the (real) frequencies at the saddle — the mode with the
imaginary frequency, the reaction coordinate, is excluded.

## Arrhenius and Eyring forms

In practice the prefactor is often taken as a constant **attempt frequency**
$\nu_0$, giving the Arrhenius form

$$
k = \nu_0 \, \exp\!\left(-\frac{\Delta E}{k_B T}\right),
\qquad \nu_0 \sim 10^{12}\text{–}10^{13}\ \text{s}^{-1},
$$

or, in the thermodynamic formulation (Eyring, 1935),

$$
k = \frac{k_B T}{h} \, \exp\!\left(-\frac{\Delta G^{\ddagger}}{k_B T}\right).
$$

The Arrhenius form is what pyKMC's `style = constant` rate computes: the
user-supplied `k0` is $\nu_0$ and the barrier comes from the event — see the
[KMC Parameters](../parameters.md) page (`[RateConstant]` section).

## From saddle searches to rates in pyKMC

pyKMC does not assume a barrier — it finds saddle points directly with
**pARTn** (ART nouveau): random activation away from a minimum, followed by
convergence to a first-order saddle using the lowest curvature mode. Each
discovered event records $\Delta E_\text{forward}$ and
$\Delta E_\text{backward}$, which feed the rate constant above. See the
[Algorithm Overview](general_algorithm.md) for where this sits in the KMC
loop.

## References

- H. Eyring, *J. Chem. Phys.* **3**, 107 (1935).
- G. H. Vineyard, *J. Phys. Chem. Solids* **3**, 121 (1957) — harmonic TST
  prefactor.
- A. F. Voter, *Introduction to the Kinetic Monte Carlo Method*, in Radiation
  Effects in Solids (Springer, 2007) — TST in the KMC context.
- G. Henkelman and H. Jónsson, *J. Chem. Phys.* **111**, 7010 (1999) — saddle
  point searches without final states.
- G. T. Barkema and N. Mousseau, *Phys. Rev. Lett.* **77**, 4358 (1996) — ART;
  pARTn documentation: <https://mammasmias.gitlab.io/artn-plugin/>.
