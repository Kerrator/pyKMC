# HTST A/B MPI smoke result

Date: June 15, 2026

The smoke ran from `/home/kerr/pykmc/pyKMC_HTST` at commit
`82489fc0565d5e970ac40069a35804dad6ed4e59`. This contains the fetched
`Kerrator/pyKMC:SK_HTST_backend` head (`4681662`) plus the two direct
`fix/htst-rate-units` follow-up commits.

## Configuration

- System: bundled 511-atom silicon vacancy example
- MPI: 8 ranks, 7 pyKMC sessions, rank 0 excluded from engine sessions
- Execution: constant followed by HTST, sequential with no overlap
- Workload: 1 KMC step, `nsearch = 1`, paired seed `19073`
- Disabled: basins and event recycling
- HTST smoke knobs: `free_radius = 4.0`, `fd_step = 0.01`, `premin = False`
- LAMMPS: version `20250722`, packages `PHONON` and `PLUGIN`; local `mtp`
  pair style retained

Command:

```bash
/home/kerr/pykmc/pykmc_env/bin/python scripts/run_htst_ab.py \
  --input /home/kerr/pykmc/docs/htst_ab_smoke_input/input.in \
  --output /home/kerr/pykmc/docs/htst_ab_smoke \
  --python /home/kerr/pykmc/pykmc_env/bin/python \
  --launcher mpirun \
  --nproc 8 \
  --steps 1 \
  --nsearch 1 \
  --repeats 1 \
  --free-radius 4.0 \
  --premin False \
  --timeout-seconds 900 \
  --overwrite
```

## Result

| Style | Launcher wall | Step wall | Events | Finite nu0 |
|---|---:|---:|---:|---:|
| constant | 6.312 s | 0.598 s | 3 | 0 |
| HTST | 6.968 s | 1.195 s | 3 | 1 |

The HTST/constant launcher wall-time ratio was `1.104`. Both runs returned zero
and generated the same number of reference events. The accepted HTST prefactor
was `17.766 THz`.

HTST coverage was only `33.3%`: two events used the configured `k0` fallback.
This is enough to confirm the MPI and online-HTST path executes, but it is not a
full-coverage performance benchmark. The launcher stderr also contains repeated
non-fatal `hwloc` topology warnings from the local kernel/hardware description.

Durable outputs:

- `/home/kerr/pykmc/docs/htst_ab_smoke/summary.json`
- `/home/kerr/pykmc/docs/htst_ab_smoke/runs.csv`
- `/home/kerr/pykmc/docs/htst_ab_smoke/repeat_001/`
