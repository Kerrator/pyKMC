"""Test helper for recording and capping dispatch without replacing worker Futures."""

import numpy as np


def recorder(mgr):
    count = {"local": 0, "group": 0}

    class Recorder:
        """Copy submitted starts before the caller reuses arrays; return the real Future."""

        def __getattr__(self, key):
            return getattr(mgr, key)

        def partn_refine(self, **kw):
            return self.submit("partn_refine", **kw)

        def group_minimize_with_results(self, **kw):
            return self.submit_group("minimize_with_results", **kw)

        def submit(self, op_name, **kw):
            if op_name == "partn_refine":
                if count["local"] == 2:
                    raise RuntimeError("pilot local refinement budget exhausted")
                count["local"] += 1
                np.savez(
                    f"local-{count['local']}.npz",
                    positions=kw["positions"],
                    ids=kw["saddle_idx"],
                    center=kw["central_atom_idx"],
                )
            return mgr.submit(op_name, **kw)

        def submit_group(self, op_name, **kw):
            if op_name == "minimize_with_results":
                if count["group"] == 3:
                    raise RuntimeError("pilot group minimization budget exhausted")
                count["group"] += 1
                if kw.get("positions") is not None:
                    np.save(f"group-{count['group']}-start.npy", kw["positions"])
            return mgr.submit_group(op_name, **kw)

    return Recorder(), count
