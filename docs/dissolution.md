# Dissolution (dealloying) events

**Status:** implemented; validated on smoke runs; production parameterization below.
This page is the design record and the *provenance of every assumption* in the
model — what comes from the literature (with quotes), what is derived from our
own potential, and what is our own modeling choice.

## What the feature does

When enabled (`[Control] dissolution = True`), an atom of a configured
less-noble species (Cr in NiCr, Fe in NiFe — never the Ni matrix) whose
first-shell (`rnei`) coordination is `n <= coord_max` competes in the
rejection-free (BKL) selection with a synthetic **dissolution event**. Selecting
it deletes the atom: the surface recedes as under-coordinated solute atoms
dissolve, which is the microscopic mechanism of dealloying. Deletions expose new
under-coordinated sites, so the active zone follows the receding surface.

## Rate law

```
k_diss(n) = nu_d * exp( (phi - n * E_b) / (kb * T) )
```

- `n` — the atom's current first-shell coordination (all first-nearest-neighbour
  bonds counted equally, one effective `E_b`),
- `E_b` — effective bond energy (eV per bond),
- `phi` — electrochemical driving force / overpotential (eV; `0` = pure bond
  counting),
- `nu_d` — attempt frequency (**ps⁻¹**, same unit contract as `k0`/`nu0`; a
  Hz-scale value is rejected by config validation).

This is exactly the canonical Erlebacher form. Erlebacher, Aziz, Karma,
Dimitrov & Sieradzki, *Nature* **410**, 450 (2001) (free full text:
[arXiv:cond-mat/0103615](https://arxiv.org/pdf/cond-mat/0103615)):

> "the dissolution rate k_E,N for a silver atom with N near neighbors was
> written as k_E,N = ν_E exp(−(Nε − φ)/k_BT), where ν_E = 10^4 sec^-1 is an
> attempt frequency determined by the exchange-current density in the BV
> [Butler–Volmer] equation and φ is the overpotential."

## Provenance of each assumption

Legend: **[LIT]** = verified literature claim (adversarially fact-checked,
vote in parentheses); **[EAM]** = derived from our own Béland 2017 NiFeCr
potential (`toolkit/dissolution/derive_Eb_eam.py`); **[OURS]** = our own
modeling choice, not literature-anchored.

| Assumption | Provenance | Source |
|---|---|---|
| Bond-counting rate, coordination-dependent, one effective `E_b` for all first-NN bonds | **[LIT]** (3–0) | Erlebacher 2001 (Nature 410, 450); Erlebacher 2004 (J. Electrochem. Soc. 151, C614): "site coordination-dependent dissolution of the less-noble atoms" |
| Only the less-noble species dissolves; the noble matrix only diffuses | **[LIT]** (3–0) | Erlebacher 2001/2004; also the Ag-Au/Co-Pd KMC of *J. Phys. Chem. C* 126 (2022): "dissolution of the less noble alloy components (Ag or Co only)" |
| Overpotential enters additively in the exponent, `+phi`, lowering the effective barrier `n*E_b − phi` | **[LIT]** (3–0) | Erlebacher 2001: `k_E,N = ν_E exp(−(Nε − φ)/kBT)`; J. Phys. Chem. C 2022: `k_diss = ν_E·exp((−nE_b + eϕ)/kBT)` |
| Dissolution attempt frequency is **not** a phonon frequency: canon uses `ν_E = 10^4 s⁻¹` (= 1e-8 ps⁻¹), anchored to the Butler–Volmer exchange-current density, 9 orders below the diffusion prefactor `ν_D = 10^13 s⁻¹` | **[LIT]** (3–0) | Erlebacher 2001; independently reused in J. Phys. Chem. C 2022 |
| `E_b(Cr–in–Ni) = 0.35 eV/bond`, `E_b(Fe–in–Ni) = 0.29 eV/bond`, `E_b(Ni–in–Ni) ≈ 0.39–0.41 eV/bond` | **[EAM]** | Slope of unrelaxed removal energy vs `n` on our production slabs (R² > 0.99); see derivation below. Same scale as Erlebacher's Au: ε = 0.285 eV/bond at 600 K (3–0) |
| Eligibility cutoff `coord_max = 6` (kinks and worse dissolve; 7-coordinated vacancy-ring atoms rearrange but do not dissolve) | **[OURS]** | **No direct canon precedent** — the claim that the Ag-Au canon uses a hard n≤6 cutoff was *refuted* (0–3) in fact-checking. The canon suppresses high-n dissolution continuously via the exponent. Our hard cutoff is a computational choice consistent with `coordination_threshold = 8` surface attention; the exponential makes it nearly equivalent (an n=7 rate would be `exp(−E_b/kT)` ≈ 3000× slower than n=6 at 500 K anyway). |
| Deleted atoms are gone (no redeposition / reverse reaction) | **[OURS]** | Canon models operate in the high-driving-force Tafel regime where redeposition is negligible (Erlebacher 2001: "consistent with the Butler-Volmer equation in the high-driving-force Tafel regime"); we inherit that regime assumption. |
| No solvent/oxide chemistry: `phi` is an *effective* driving-force knob absorbing solvation, redox and the EAM embedding offset — not a literal electrode potential | **[OURS]** | See "What phi means" below. |

## Deriving E_b from our own potential

`toolkit/dissolution/derive_Eb_eam.py` (run it to reproduce): take the
production 1-vacancy slabs, pick 4 well-separated clean terrace atoms (n = 8)
of the target species, strip their neighbours in-plane-first to walk the ladder
n = 8 → 3, and at each rung compute the **unrelaxed** removal energy
`dE(n) = E(without target) − E(with target)` with single-point `eam/alloy`
evaluations (alphabetical type order, identical to the engine convention).
Least-squares fit `dE(n) = E_b·n + c` pooled over targets.

Unrelaxed by design: bond counting is a rigid-lattice concept, and relaxation
and solvation effects belong in `phi`, not in `E_b`.

Results (Béland 2017 NiFeCr EAM, `NiFeCr_LKB2017.eam`):

| System | Species removed | E_b (eV/bond) | intercept c (eV) | R² |
|---|---|---|---|---|
| Ni₉₅Cr₀₅ slab | **Cr** | **0.350** | +1.28 | 0.992 |
| Ni₉₅Cr₀₅ slab | Ni | 0.410 | +2.03 | 0.992 |
| Ni₉₅Fe₀₅ slab | **Fe** | **0.289** | +2.88 | 0.994 |
| Ni₉₅Fe₀₅ slab | Ni | 0.390 | +2.27 | 0.989 |

Two physics checks pass: (i) removal energy is genuinely linear in `n`
(R² > 0.99), so bond counting is a good model for this EAM; (ii) the solute is
cheaper to strip per bond than Ni in both alloys — the selectivity that drives
dealloying is present in the potential itself.

The intercept `c` (the EAM many-body/embedding offset) is *constant across n*,
so it is exactly degenerate with `phi` and is deliberately **not** put into the
rate: `phi` absorbs it along with all solution chemistry.

## What phi means (and what it does not)

`phi` is the single knob for the environment's driving force. The literature
organizes dealloying behavior around the **critical potential**: below it,
surfaces passivate; above it, sustained dissolution/porosity (Erlebacher 2004,
3–0: "an intrinsic critical potential exists as a well-defined threshold
potential separating surface passivation and porosity formation behaviors").
In our rate the same threshold appears at

```
phi_c ~ coord_max * E_b        (k_diss(coord_max) = nu_d at phi = phi_c)
```

- **NiCr:** `phi_c ≈ 6 × 0.350 = 2.10 eV`
- **NiFe:** `phi_c ≈ 6 × 0.289 = 1.73 eV`

For scale, Erlebacher's published Ag-Au simulations ran at φ = 1.75–1.8 eV with
ε = 0.285 eV (φ/6ε ≈ 1.02–1.05 — just supercritical).

Because `phi` and `T` are independent inputs, a temperature sweep at fixed
`phi` is an *iso-potential* sweep — the physically meaningful experiment.
(Folding the driving force into `nu_d` instead would require a
temperature-dependent, astronomically large prefactor, which the units guard
correctly rejects.)

## Recommended starting parameters

```ini
[Control]
dissolution = True

[Dissolution]
elements = Cr          # Fe for the NiFe system
nu_d = 1e-8            # ps^-1  == 1e4 s^-1, the Butler-Volmer-anchored canon value
E_b = 0.35             # eV/bond, EAM-derived (0.289 for Fe)
coord_max = 6
phi = 2.10             # eV: at the intrinsic critical potential (1.73 for NiFe)
```

Worked rates at T = 500 K, NiCr at `phi = phi_c = 2.10`:

| n | k_diss | vs. a 0.6 eV surface hop (ν₀=10 THz → ~9×10⁶ s⁻¹) |
|---|---|---|
| 6 (kink) | 1×10⁴ s⁻¹ | rare (~10⁻⁴ of one hop) |
| 5 | 3.4×10⁷ s⁻¹ | competitive — dissolves within a few KMC steps |
| 4 (adatom-like) | 1.1×10¹¹ s⁻¹ | immediate |
| 3 | 3.8×10¹⁴ s⁻¹ | immediate |

Sweep suggestion: `phi ∈ {phi_c − 0.2, phi_c, phi_c + 0.2}` spans
passivation-like → onset → driven dealloying without any other change.

## What a *correct* model must show at 5 at% solute

Our compositions sit **far below every measured parting limit**, and that is a
feature of the validation, not a bug of the model:

- Aqueous fcc canon: parting limit 50–60 at% reactive element; Ag-Au ≈ 55 at%
  (Artymowicz, Erlebacher & Newman, *Phil. Mag.* 2009, 3–0), set by a
  high-density site-percolation threshold (pc(9) = 59.97 ± 0.03 %, 3–0).
- Molten chlorides (Ghaznavi, Persaud & Newman, *J. Electrochem. Soc.* 2022 +
  Ghaznavi thesis, U. Toronto): parting limit ≈ 38 at% at 350 °C, < 32 at% at
  600 °C, ≈ 22 at% at 700 °C (3–0) — and the same percolation/surface-diffusion
  mechanism as aqueous dealloying governs (3–0).
- FCC 3rd-NN percolation threshold: 6.1 at% (Nature Materials **20** (2021),
  2–0) — 5 at% sits just *below* even that most permissive connectivity bound.

So at Ni₉₅X₀₅ the correct behavior is **surface-limited dissolution with
Ni-enrichment slowdown**: the exposed solute inventory dissolves, the surface
Ni-enriches, the dissolution rate decays — no sustained bulk porosity. If a run
shows runaway bulk dealloying at 5 at%, the parameters (most likely `phi`) are
unphysically high. Observables to track per run: cumulative dissolved-atom
count vs time (the dealloying curve, logged at each `[dissolution]` event),
remaining-solute depth profile, and the coordination histogram of dissolved
atoms.

## Known scope limits (documented honestly)

1. **Ni never dissolves in our model.** Below the parting limit in molten
   chloride experiments, *both* Ni and Cr dissolve with uniform surface
   recession (Cr22Ni78 at 500 °C — Ghaznavi thesis, 2–1). Our solute-only
   deletion cannot reproduce Ni codissolution; it models the selective-attack
   channel only.
2. **Aqueous NiCr may invert the selectivity.** One (unverified, 1–0 with 2
   errored votes) claim from the Nature Materials 2021 work: in 0.1 M H₂SO₄,
   online ICP-MS shows *Ni* dissolving selectively while Cr passivates as
   oxide. Our chemistry-free model maps better onto molten-salt-like
   environments than onto aqueous passivating ones.
3. **Refuted transfer rule.** Scaling E_b between alloys by melting-point
   ratio (as claimed for Ag-Au → Co-Pd) failed fact-checking (1–2). We do not
   transfer E_b; we derive it from our own EAM per system.
4. **DFT cross-check, molten salt:** Cr dissolution barriers from Ni-Cr into
   FLiNaK are 2.4–2.7 eV (arXiv:2510.25098, 3–0). Our zero-phi kink barrier
   `6 × 0.35 = 2.10 eV` is the same order — consistent, but our barrier
   excludes solvation, so the numbers are not directly comparable.
5. A vacancy can strand one layer below the surface (invisible at
   `coordination_threshold = 8`); accepted approximation — count strandings
   per trajectory when analyzing.

## References

1. J. Erlebacher, M. J. Aziz, A. Karma, N. Dimitrov, K. Sieradzki, *Evolution
   of nanoporosity in dealloying*, Nature **410**, 450 (2001).
   [arXiv:cond-mat/0103615](https://arxiv.org/pdf/cond-mat/0103615)
2. J. Erlebacher, *An atomistic description of dealloying: porosity evolution,
   the critical potential, and rate-limiting behavior*, J. Electrochem. Soc.
   **151**, C614 (2004). [doi:10.1149/1.1784820](https://iopscience.iop.org/article/10.1149/1.1784820)
3. D. A. Artymowicz, J. Erlebacher, R. C. Newman, *Relationship between the
   parting limit for de-alloying and a particular geometric high-density site
   percolation threshold*, Phil. Mag. (2009).
   [doi:10.1080/14786430903025708](https://www.tandfonline.com/doi/abs/10.1080/14786430903025708)
4. Ag-Au / Co-Pd dealloying KMC with explicit ν_D/ν_E split, J. Phys. Chem. C
   **126** (2022). [post-print](https://arts.units.it/bitstream/11368/3028745/3/acs.jpcc.1c09592-Post_print.pdf)
5. M. Ghaznavi, S. Y. Persaud, R. C. Newman, *Electrochemical corrosion
   studies in molten chloride salts*, J. Electrochem. Soc. **169** (2022).
   [doi:10.1149/1945-7111/ac735b](https://iopscience.iop.org/article/10.1149/1945-7111/ac735b);
   thesis: [U. Toronto](https://utoronto.scholaris.ca/server/api/core/bitstreams/a5841dd2-487c-42ba-81c3-6f294bfb6c1c/content)
6. K. Sieradzki et al. (percolation design of corrosion-resistant alloys),
   Nature Materials **20** (2021).
   [PDF](https://engineering.jhu.edu/mtaheri/MURI/wp-content/uploads/2022/10/s41563-021-00920-9.pdf)
7. Cr dissolution into FLiNaK, DFT barriers for ordered/disordered Ni-Cr,
   [arXiv:2510.25098](https://arxiv.org/html/2510.25098)
8. KMC dealloying of Pt-alloy nanoparticles under applied potential,
   Electrochim. Acta (2013).
   [doi:10.1016/j.electacta.2013.01.053](https://www.sciencedirect.com/science/article/abs/pii/S0013468613000856)

*Fact-checking provenance: all bracketed votes are from an adversarial
verification pass (3 independent refutation attempts per claim; a claim dies on
2/3 refutes) run 2026-07-17 over 17 fetched sources / 69 extracted claims → 20
confirmed, 3 refuted, 2 unverified.*
