"""Build the LAMMPS-free replay fixtures from issue 02's pARTn recordings.

Search 0 keeps all evaluations and the saved endpoints. Searches 1–7
keep the initial pushes and the first Lanczos probe. Run from any directory:

    python -I make_replay_set.py SRC DST
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

LANC = 5  # pARTn's disp_code for a Lanczos move
PER_EVAL = (
    "x",
    "f",
    "v_in",
    "scalars_in",
    "scalars_out",
    "partn_step",
    "repeat",
    "kind",
)
DRAWS = ("draw_initial_push", "draw_first_lanczos_vector")
SYSTEM = ("movable", "cell", "pbc")
ENDPOINTS = (
    "partn_tau_init",
    "partn_tau_sad",
    "partn_tau_min1",
    "partn_tau_min2",
    "partn_eigen_sad",
)

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for search in range(8):
    name = f"search{search}.npz"
    with np.load(src / name, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        steps = z["partn_step"]
        if search == 0:
            end = len(steps) - 1
            keys = (
                *PER_EVAL,
                *DRAWS,
                "draw_new_chance_push",
                *SYSTEM,
                "eigenvec_index",
                "eigenvec",
                *ENDPOINTS,
            )
            keep = {key: z[key] for key in keys}
        else:
            codes = z["scalars_out"][:, meta["scalars"].index("partn_disp_code")]
            numbered = np.flatnonzero(steps >= 0)
            # Step k + 1 is led by the move selected on numbered step k.
            first_probe = int(np.flatnonzero(codes[numbered[:-1]] == LANC)[0]) + 1
            end = int(numbered[first_probe])
            keep = {key: z[key][: end + 1] for key in PER_EVAL}
            keep.update({key: z[key] for key in (*DRAWS, *SYSTEM)})
            meta["trimmed"] = (
                f"evaluations 0-{end} of {len(steps)} (issue 04 initial-push window)"
            )
        text = json.dumps(meta)
        if search == 0:
            # The recording includes local paths in its engine commands and provenance.
            text = re.sub(
                r'/(?:Users|home)/[^"\s]+', lambda match: Path(match[0]).name, text
            )
        else:
            # Preserve the original prefix metadata byte for byte.
            text = re.sub(r'/(Users|home)/[^/"]+', "~", text)
        keep["meta_json"] = np.array(text)
    out = dst / name
    np.savez_compressed(out, **keep)
    print(f"{name}: evaluations 0-{end}, {out.stat().st_size} bytes")
