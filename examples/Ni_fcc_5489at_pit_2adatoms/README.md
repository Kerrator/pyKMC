# Ni_fcc_5489at_pit_2adatoms — two adatoms in a (100) pit: basin mapping in pure Ni and in NiFe

A basin-acceleration test whose transient states are *known by construction*: two adatoms random-walk on the
floor of a square pit until one reaches the pit wall or the two meet and form a dimer. Each sub-directory is a
self-contained example (copy `input.in.template` to `input.in`, then `mpirun -n 8 python -m pykmc -in input.in`).

| sub-example | system | what it tests |
|---|---|---|
| `Ni/` | 5,489 Ni atoms | the reference: the basin is the Ni-adatom walk, exits = wall attachment or dimer |
| `Ni_oneFe/` | same, one adatom is Fe | a species-coloured basin with one immobile trap (the Fe adatom) |
| `Ni95Fe05/` | same, 275 Fe (5 at.%) through the slab, 23 in the floor layer | the full-colour catalogue explosion: every floor site is a different chemical environment |

Geometry (all three): 16×16 fcc(100) slab, a = 3.52 Å, 11 layers, 40 Å vacuum. The top layer has a 12×12-site pit
(`role = floor` marks the 144 exposed atoms of the layer below, z = 15.84 Å); the 76 top-layer atoms lining the
pit are `role = wall` (indices 5411–5486); the two `role = adatom` atoms (5487, 5488) sit on hollow sites of
the pit floor, 8.97 Å from the nearest wall atom and 19.9 Å apart. Potential: `../NiFeCr_LKB2017.eam`
(Béland 2017 / Bonny 2011 Ni–Fe–Cr), `pair_coeff` elements listed alphabetically (`Fe Ni`).

## Barriers (LKB2017, 300 K, measured at step 1)

| event | forward / backward (eV) | role in the basin |
|---|---|---|
| Ni adatom hop, hollow → hollow | 0.539 / 0.539 | the flicker (transient) |
| Fe adatom hop | 0.913 / 0.913 | 1.7×10⁶ slower than Ni at 300 K → the Fe adatom is a static trap |
| Fe adatom "shoulder" (metastable, +0.516 eV) | 0.517 / 0.001 | transient, stiff (returns at ~k0) |
| wall-ring atom detaching onto the floor | 0.710 / 0.272 | absorbing — excluded via `[inactive_atoms]`, see below |
| wall-ring atom sliding along the edge | 0.743 / 0.740 | absorbing — excluded via `[inactive_atoms]` |
| adatom ↔ wall attachment, dimer formation | learned during the run | the intended exits (backward ≫ `energy_thr`) |

## Why the template looks the way it does

- `[IRA] sym_thr = 0.05` (default 0.01). Reference events are stored once and their direction multiplicity comes
  only from the symmetry operations SOFI finds in the relaxed rcut shell. At 0.01 Å the relaxed adatom shell
  yields identity + one mirror (1 hop direction: a 1-D walk); at 0.05 all 8 C4v operations (4 directions).
  Check `len(sym_matrix)` in `reference_table.pickle` after step 1 — it must be 4 for the hop.
- `rcut = 8.0`, `rnei = 3.2`. The relaxed adatom's distance spectrum has shells at 8.48 and 8.55 Å — the usual
  8.5 Å cut-off sits between them with 0.02 Å margins, so shell membership (and IRA matching) flips from site to
  site; 8.0 Å sits in a 0.3 Å gap. The relaxed adatom–floor bond is 2.96 Å, too close to `rnei = 3.0`.
- `[inactive_atoms] indices = 5411 … 5486` (the wall ring). Without it every transient state re-lists the wall
  atoms' 0.71/0.74 eV events as absorbing exits and each row costs a pARTn refinement: 22 transient states
  became 1,626 absorbing states and 30 min of wall time. With the ring inactive the basin contains only adatom
  hops and its exits are the physical ones (adatom at the wall, dimer). The wall atoms are still classified
  (their ids count as visited), just never used as event centres. `coordination_threshold = 7` is *not* a
  substitute: 32 of the 76 ring atoms have coordination 6.
- `energy_thr = 0.62`: above the 0.539 eV hop, below every wall/attachment barrier, ≥ 0.08 eV from both.
- `[Reconstruction] mover_landing_fraction = 0.35` (opt-in): lands the pushed mover inside the barrier on
  asymmetric saddles (the attachment events are asymmetric).
- `max_states = 400`, `max_basin_walltime_s = 1800`: a *learning-phase* budget. Remove both (None = unlimited)
  for a full-basin map once the catalogue has converged (see below).

## Measured behaviour (2026-08-27 scoping runs, 7 ranks, `SK_alloy_basins_fix` @ 31ab8b8)

Learning phase (pure Ni, `basin = False`, 60 steps, 9 min): the catalogue converged — 0 new environments
after step ~31; the first wall attachment happened at step 22, after which the run cycles detach (1.15 eV) /
re-attach (0.28 eV). The seeded full-map run points `[Control] reference_table` / `visited_environments` at
that run's pickles and removes `max_states`.

| run | steps | basins | largest basin (transient + absorbing) | exits accepted | acceleration | verdict |
|---|---|---|---|---|---|---|
| `Ni` seeded full map (no state cap, 12 h walltime) | 3 | 2 | **2,817 + 7,183** (22,830 rows; 12,831 duplicate merges; the walltime cap deferred 5,667 frontier states) | 2/2 | 11× — exit = wall attachment (0.28 eV) | mappable; not yet mapped to the last state |
| `Ni_oneFe` | 40 | 6 | 51 + 241 | 6/6 | up to 20× | Ni-adatom basin maps; the Fe adatom is its own 0.91 eV walker |
| `Ni95Fe05` | 40 | 40 | 40 + ~154 | 40/40 | median ≈ 1×, max 14× | **not mappable at 5 at.%** — every basin is 100 % unknown-environment frontier, still learning 1–5 ids/step at step 40 |

The pure-Ni story completes physically in 3 KMC steps: basin 1 → first adatom attaches to the wall, basin 2
(87 + 380, 10 min) → second adatom attaches, step 3 = plain-KMC detachment (1.148 eV, Δt = 1.4×10⁶ s).
Honest caveats: (i) in the 12 h map, 72 % of the absorbing flux still sat on the deferred frontier — those
states are really transient, so t_exit is biased low until the frontier is closed (more walltime, or the
wavefront strategy with `n_workers`); (ii) no run ever formed a dimer (minimum adatom separation 6.6 Å in
pure Ni), so the dimer exit is designed but unexercised — start the adatoms closer to test it; (iii) all
48 basin exits across the three final runs reconstructed on the first try with
`mover_landing_fraction = 0.35` and `emin_event = 0.02` (the v4 alloy run without the emin filter lost 33/39
exits to a 0.5 meV pARTn shelf state of the Fe adatom).

## Mapping the full basin

Two things limit completeness, and they are different in the pure and alloy cases:

1. **Unknown-environment truncation.** A basin state whose atoms carry an environment id the main loop has never
   searched is made absorbing (the exits of that state are unknown). An adatom that steps within `rcut` of the
   wall — or of the other adatom — gets a new id, so the first basins stop one hop short of the wall. The main
   loop then searches those environments (≈4 per step) and the next basin goes further. In pure Ni the set is
   finite (distance-to-wall classes × corner/edge × adatom–adatom proximity) and converges in a few steps; in
   Ni95Fe05 every floor site is a distinct decoration, so the catalogue converges only after the adatoms have
   *visited every site at flicker speed* — the basin never accelerates anything at this concentration.
2. **State count.** Pure Ni: ~80 interior hollow sites per adatom → ~3,000 unordered two-adatom states, each
   needing 8 reconstructions (~2 s each) → ~15 h serial for one complete map. Ni_oneFe: the Fe adatom is
   immobile, so the basin is the ~80 single-adatom states — minutes.

To map the full pure-Ni basin: run the learning phase (this template, ~10 steps), then start a second run
from the converged catalogue with the caps removed, pointing `[Control] reference_table` and
`visited_environments` at the first run's `reference_table.pickle` / `visited_environments.pickle`.
