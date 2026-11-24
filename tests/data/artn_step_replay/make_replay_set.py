"""Cut issue 02's pARTn step recordings to the initial-push window.

Keeps only what tests/artn_replay.py's load_recording reads, for the
evaluations up to the first Lanczos probe, in the same layout, so the cut
files load and replay unchanged. Home folders in the metadata become ``~``.

    python -I make_replay_set.py SRC DST
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
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
ONCE = ("draw_initial_push", "draw_first_lanczos_vector", "movable", "cell", "pbc")
for path in sorted(src.glob("search*.npz")):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cols = meta["scalars"]
    codes = z["scalars_out"][:, cols.index("partn_disp_code")]
    steps = np.flatnonzero(z["partn_step"] >= 0)
    first_lanc = int(np.flatnonzero(codes[steps] == 5)[0])  # step that writes LANC
    end = int(steps[first_lanc + 1])  # the evaluation that move led to
    keep = {k: z[k][: end + 1] for k in PER_EVAL}
    keep.update({k: z[k] for k in ONCE})
    meta["trimmed"] = (
        f"evaluations 0-{end} of {len(z['partn_step'])} (issue 04 initial-push window)"
    )
    text = re.sub(r"/(Users|home)/[^/\"]+", "~", json.dumps(meta))
    keep["meta_json"] = np.array(text)
    out = dst / path.name
    np.savez_compressed(out, **keep)
    print(f"{path.name}: evaluations 0-{end}, {out.stat().st_size / 1e6:.2f} MB")
