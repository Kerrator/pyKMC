"""Cut issue 02's pARTn step recordings to the replay set in this folder.

Keeps only what tests/artn_replay.py's load_recording reads, in the same
layout, so the cut files load and replay unchanged. Each search is cut at the
first Lanczos probe (the initial-push window); the searches in
``TO_FIRST_BLOCK`` are cut at the first Lanczos block's move back instead and
keep the eigenvector versions in force up to there. Home folders in the
metadata become ``~``.

    python -I make_replay_set.py SRC DST
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

# Searches cut at the end of the first Lanczos block (Stephen, 2026-11-24).
TO_FIRST_BLOCK = {"search0.npz"}
LANC = 5  # pARTn's disp_code for a Lanczos move

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
    led_lanc = codes[steps[:-1]] == LANC  # step k + 1 was led by a Lanczos move
    last = int(np.flatnonzero(led_lanc)[0]) + 1  # the first probe
    whole_block = path.name in TO_FIRST_BLOCK
    if whole_block:
        while last + 1 < len(steps) and led_lanc[last]:
            last += 1  # through the probes to the move back
    end = int(steps[last])
    keep = {k: z[k][: end + 1] for k in PER_EVAL}
    keep.update({k: z[k] for k in ONCE})
    window = "issue 04 initial-push window"
    if whole_block:
        keep["eigenvec_index"] = z["eigenvec_index"][: end + 1]
        keep["eigenvec"] = z["eigenvec"][: int(keep["eigenvec_index"].max()) + 1]
        window = "to the first Lanczos block's move back, issue 05"
    meta["trimmed"] = f"evaluations 0-{end} of {len(z['partn_step'])} ({window})"
    text = re.sub(r"/(Users|home)/[^/\"]+", "~", json.dumps(meta))
    keep["meta_json"] = np.array(text)
    out = dst / path.name
    np.savez_compressed(out, **keep)
    print(f"{path.name}: evaluations 0-{end}, {out.stat().st_size / 1e6:.2f} MB")
