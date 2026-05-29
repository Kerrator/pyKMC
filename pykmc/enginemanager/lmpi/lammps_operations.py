import numpy as np
from ase.data import atomic_numbers, atomic_masses
from mpi4py import MPI
import ctypes
import pypARTn
from pathlib import Path
from types import SimpleNamespace
from ...utils.io_utils import capture_output
from ...utils.geometry import compute_delr_l2, count_moved_atoms
from ...activevolume.active_volume import (
    reset,
    redefine_atoms,
    partn_search_AV,
    partn_refine_AV,
    position_results_AV,
)
from ...atomic_environment import AtomicEnvironment
from ...otfml import (
    OTFML_MAX_FLAG,
    OTFML_TOL_FLAG,
    OTFML_LATCH,
    session_dump_path,
    read_otf_thermo,
    otf_thermo_path,
    OTFExtrapolationFlags,
)

from ...result import (
    Result,
    ErrorInfo,
    EventSearchOutput,
    Ok,
    Err,
    ErrorType,
    EventRefinementOutput,
)


def initialize_parameters(engine):
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary p p p")
    engine.command("atom_modify map array")  #! necessary for scatter atoms
    engine.command("atom_modify sort 0 0.0")  #! necessary for partn


def initialize_system(engine, system, config=None):

    # system parameters
    natoms = len(system.types)
    cell = system.cell
    types = system.types
    x = system.positions.flatten()  # Lammps format

    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

    ind = np.linspace(0, natoms - 1, natoms).astype(int)
    ind += 1  # Lammps id start at 1

    type_order = (
        config.lammps.type_order
        if (config and config.lammps.type_order)
        else list(dict.fromkeys(types))
    )
    if set(type_order) != set(types):
        raise ValueError(
            f"type_order {type_order} does not match the elements in the system {sorted(set(types))}"
        )
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(type_order)
    }
    types = [map_type[element]["ref"] for element in types]  # map to integer

    # lammps create system
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    engine.lmp.create_atoms(natoms, ind, types, x)
    # Set masses
    for key in map_type.keys():
        engine.command("mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"]))
    # Label atoms name to type :
    engine.command(
        "labelmap atom "
        + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
    )
    engine.type_labels = {v["ref"]: key for key, v in map_type.items()}


def initialize_potential(engine, config):
    pair_style = config.lammps.pair_style
    pair_coeff = config.lammps.pair_coeff
    engine.command("pair_style {}".format(pair_style))
    engine.command("pair_coeff {}".format(pair_coeff))
    for cmd in config.lammps.setup_commands or []:
        engine.command(cmd)

    if pair_style.strip().startswith("mtp/extrapolation"):
        try:
            engine.command(
                "fix extrapolation_grade all pair 1 mtp/extrapolation extrapolation 1"
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "Please use the MLIP-3 style extrapolation for configuration mode MTPs"
                in msg
            ):
                raise RuntimeError(
                    "The loaded MTP is in configuration mode. "
                    "Current pyKMC OTFML expects neighborhood-mode `mtp/extrapolation` "
                    "with per-atom `f_extrapolation_grade` support. "
                    "Use a neighborhood-mode MTP or disable OTFML"
                ) from exc
            raise

    if config.control.otfml:
        if not pair_style.strip().startswith("mtp/extrapolation"):
            raise RuntimeError("OTFML requires `pair_style mtp/extrapolation`.")
        gamma_tol = config.otfml.gamma_tolerance
        gamma_max = config.otfml.gamma_max
        dump_path = session_dump_path(engine.engine_id).as_posix()
        engine.command(f"variable {OTFML_TOL_FLAG} internal 0")
        engine.command(f"variable {OTFML_MAX_FLAG} internal 0")
        engine.command(f"compute max_grade all reduce max f_extrapolation_grade")
        engine.command(f"variable max_grade equal c_max_grade")
        engine.command(f'variable dump_skip equal "v_max_grade < {gamma_tol:.4f}"')
        engine.command(
            f"dump extrapolative_structures_dump all custom 1 {dump_path} id type x y z f_extrapolation_grade"
        )
        engine.command(f"dump_modify extrapolative_structures_dump append yes")
        engine.command(f"dump_modify extrapolative_structures_dump skip v_dump_skip")
        engine.command(
            f"fix extreme_extrapolation all halt 1 v_max_grade > {gamma_max:.4f} error continue"
        )
        _setup_otf_latch(engine, gamma_tol, gamma_max)
        engine.command(f"log {otf_thermo_path(engine).as_posix()}")
        # engine.command("echo none")
        engine.command(f"thermo 1")
        engine.command(
            f"thermo_style custom step pe v_max_grade v_{OTFML_LATCH} v_{OTFML_TOL_FLAG} v_{OTFML_MAX_FLAG}"
        )
        engine.command("thermo_modify line yaml flush no")


def _setup_otf_latch(engine, gamma_tol: float, gamma_max: float) -> None:
    """Register a python-style variable that latches the OTF flag internal variables.

    Evaluated every minimization step via thermo_style, so flags are updated
    the moment grade crosses either threshold — not just at the final step.
    """
    latch_code = (
        f"def _latch_otf_flags(handle, max_grade):\n"
        f"    from lammps import lammps\n"
        f"    lmp = lammps(ptr=handle)\n"
        f"    if max_grade >= {gamma_tol:.4f}: lmp.set_internal_variable('{OTFML_TOL_FLAG}', 1.0)\n"
        f"    if max_grade >= {gamma_max:.4f}: lmp.set_internal_variable('{OTFML_MAX_FLAG}', 1.0)\n"
        f"    return lmp.extract_variable('{OTFML_TOL_FLAG}')\n"
    )
    engine.command(
        f"python _latch_otf_flags"
        f" input 2 SELF v_max_grade"
        f" return v_{OTFML_LATCH}"
        f" format pff"
        f' here """{latch_code}"""'
    )
    engine.command(f"variable {OTFML_LATCH} python _latch_otf_flags")


def reload_potential(engine, config):
    """Reload an updated potential without rebuilding the LAMMPS system."""
    engine.command("pair_style {}".format(config.lammps.pair_style))
    engine.command("run 0")


def reset_otf_flags(engine) -> None:
    """Reset the latched OTF extrapolation flags on the current engine."""
    engine.command(f"log {otf_thermo_path(engine).as_posix()}")
    if not hasattr(engine.lmp, "set_internal_variable"):
        raise RuntimeError(
            "LAMMPS Python module does not expose set_internal_variable(), "
            "which is required for OTFML flag handling."
        )
    if engine.lmp.set_internal_variable(OTFML_TOL_FLAG, 0.0) != 0:
        raise RuntimeError(f"Failed to reset OTFML variable '{OTFML_TOL_FLAG}'.")
    if engine.lmp.set_internal_variable(OTFML_MAX_FLAG, 0.0) != 0:
        raise RuntimeError(f"Failed to reset OTFML variable '{OTFML_MAX_FLAG}'.")


def get_thermo_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read OTF flags from the already-parsed engine.otf_thermo block."""
    # Read last thermo YAML block for diagnostics only.
    engine.command("log none")  # force flush
    otf_thermo = read_otf_thermo(engine)

    if otf_thermo is None:
        raise RuntimeError("OTF thermo data not available on engine.")

    tol_col = f"v_{OTFML_TOL_FLAG}"
    max_col = f"v_{OTFML_MAX_FLAG}"

    return OTFExtrapolationFlags(
        extrapolated=bool(max(otf_thermo[tol_col]) > 0),
        extreme_extrapolated=bool(max(otf_thermo[max_col]) > 0),
    )


def get_lammps_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read the current latched OTF extrapolation flags from the engine."""

    def extract_scalar(name: str) -> float:
        # try:
        value = engine.lmp.extract_variable(name, None, 0)
        # except TypeError:
        #     value = engine.lmp.extract_variable(name)
        return float(value)

    return OTFExtrapolationFlags(
        extrapolated=bool(extract_scalar(OTFML_TOL_FLAG)),
        extreme_extrapolated=bool(extract_scalar(OTFML_MAX_FLAG)),
    )


def get_otf_flags(engine) -> OTFExtrapolationFlags:
    """Read the current latched OTF extrapolation flags from the engine."""

    return get_lammps_otf_flags(engine)
    # return get_thermo_otf_flags(engine)


def _build_extrapolation_error(
    flags: OTFExtrapolationFlags,
    *,
    phase: str,
    message: str,
    variables: dict,
):
    if flags.extreme_extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTREME_EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    if flags.extrapolated:
        return Err(
            ErrorInfo(
                type=ErrorType.EXTRAPOLATION,
                message=message,
                variables={"phase": phase, **variables},
            )
        )
    return None


def minimize(engine, config, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))


def get_total_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # Get total energy
    engine.command("run 0")
    result = engine.lmp.get_thermo("etotal")
    if engine.rank == 0:
        return result


def get_potential_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # get potential energy
    engine.command("compute c1 all pe")
    engine.command("run 0")
    result = engine.lmp.extract_compute("c1", 0, 0)
    engine.command("uncompute c1")
    return result


def get_positions(engine) -> np.ndarray:
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        result = np.ctypeslib.as_array(result)
        result = np.reshape(result, (-1, 3))
    else:
        result = None
    return engine.engine_comm.bcast(result, root=0)


def get_types(engine) -> list[str]:
    int_types = engine.lmp.gather_atoms("type", 0, 1)
    if engine.rank == 0:
        types = [engine.type_labels[t] for t in int_types]
    else:
        types = None
    return engine.engine_comm.bcast(types, root=0)


def set_positions(engine, positions):
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)


def minimize_with_results(engine, config, positions=None, types=None):
    """
    Minimize and return the minimized positions and the total energy.
    """
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    atoms_frozen = _make_frozen_group(engine, config, positions, types)
    _apply_frozen_fix(engine, "f_frozen_min", atoms_frozen)
    minimize(engine, config)
    _remove_frozen_fix(engine, "f_frozen_min", atoms_frozen)
    _delete_frozen_group(engine, atoms_frozen)
    new_positions = get_positions(engine)
    potential_energy = get_potential_energy(engine)
    if engine.rank == 0:
        return new_positions, potential_energy


def minimize_freeze_core(
    engine, central_atom_positions: np.ndarray, rcut: float, maxiter: int = 10
):
    """
    Minimize with fix atom around central atom up to rcut
    """

    # define core region and group
    engine.command(
        f"region sphere_region sphere {central_atom_positions[0]} {central_atom_positions[1]} {central_atom_positions[2]} {rcut}"
    )
    engine.command("group frozen_group region sphere_region")

    # freeze core region
    engine.command("fix freeze frozen_group setforce 0.0 0.0 0.0")

    # minimization
    engine.command(f"min_style cg")
    engine.command(f"minimize 1e-6 1e-8 {maxiter} {maxiter}")

    # unfreeze/delte
    engine.command("unfix freeze")
    engine.command("group frozen_group delete")
    engine.command("region sphere_region delete")


def _make_frozen_group(engine, config, positions, types) -> bool:
    """Resolve frozen atoms and create g_frozen group. Returns True if any atoms are frozen."""
    if config.frozen_atoms is None:
        return False
    if positions is None:
        positions = get_positions(engine)
    if types is None:
        types = get_types(engine)
    frozen_ae = AtomicEnvironment(
        style="region",
        region=config.frozen_atoms,
        positions=positions,
        atom_types=types,
    )
    frozen_indices = frozen_ae.get_atoms_with_id("in")
    if not frozen_indices:
        return False
    lammps_ids = " ".join(str(i + 1) for i in frozen_indices)  # 1-based LAMMPS IDs
    engine.command(f"group g_frozen id {lammps_ids}")
    return True


def _apply_frozen_fix(engine, fix_name: str, atoms_frozen: bool) -> None:
    """Add a setforce 0 0 0 fix on g_frozen under the given name."""
    if atoms_frozen:
        engine.command(f"fix {fix_name} g_frozen setforce 0.0 0.0 0.0")


def _remove_frozen_fix(engine, fix_name: str, atoms_frozen: bool) -> None:
    """Remove a setforce fix previously added by _apply_frozen_fix."""
    if atoms_frozen:
        engine.command(f"unfix {fix_name}")


def _delete_frozen_group(engine, atoms_frozen: bool) -> None:
    """Delete the g_frozen group after all fixes on it have been removed."""
    if atoms_frozen:
        engine.command("group g_frozen delete")


def _reset_engine_state(engine, config, positions, cell, types) -> None:
    """Rebuild the local LAMMPS state after a search crashes mid-command."""
    # if positions is None or cell is None or types is None:
    #     return
    system = SimpleNamespace(
        types=np.array(types, copy=True),
        positions=np.array(positions, copy=True),
        cell=np.array(cell, copy=True),
    )
    engine.command("clear")
    initialize_parameters(engine)
    initialize_system(engine, system, config)
    initialize_potential(engine, config)


@capture_output()
def partn_search(
    engine, config, central_atom_idx: int, positions=None, cell=None, types=None
):
    try:
        # Check to see if system is in AV mode:
        if config.control.active_volume == True:
            atom_map, central_lammps_id = partn_search_AV(
                engine, config, central_atom_idx, positions, cell, types
            )

        else:
            # Set positions
            atom_map = None
            central_lammps_id = [central_atom_idx + 1]
            if positions is not None:
                set_positions(engine=engine, positions=positions)

        # INITILIZE ARTN on all ranks
        artn = pypARTn.artn(engine="lammps")

        # LAMMPS COMMANDS
        engine.command(f"plugin load {artn.lib._name}")
        atoms_frozen = _make_frozen_group(engine, config, positions, types)
        _apply_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
        engine.command("fix 10 all artn dmax {}".format(config.partn.dmax))
        _apply_frozen_fix(engine, "f_frozen_post", atoms_frozen)
        engine.command("min_style fire")

        # SETUP ARTN
        artn.reset_input()
        # Control
        artn.set("filout", "artn.out." + str(engine.engine_id))
        artn.set("engine_units", "lammps/metal")
        artn.set("verbose", config.partn.verbosity)
        artn.set("struc_format_out", "none")
        artn.set("delr_thr", config.partn.delr_thr)

        # Exploration
        artn.set("lpush_final", True)
        artn.set(
            "lmove_nextmin", False
        )  # if true fortran runtime error when event not found
        artn.set("zseed", config.partn.zseed)

        # Initial push
        artn.set("push_mode", config.partn.push_mode)
        if config.partn.push_mode == "rad":
            artn.set("push_dist_thr", config.partn.push_dist_thr)
        artn.set("push_step_size", config.partn.push_step_size)
        artn.set("push_ids", central_lammps_id)
        artn.set("ninit", config.partn.ninit)

        # Lanczos
        artn.set("lanczos_min_size", config.partn.lanczos_min_size)
        artn.set("lanczos_max_size", config.partn.lanczos_max_size)
        artn.set("lanczos_disp", config.partn.lanczos_disp)
        artn.set("lanczos_eval_conv_thr", config.partn.lanczos_eval_conv_thr)

        # Eigenvector push
        artn.set("eigval_thr", config.partn.eigval_thr)
        artn.set("eigen_step_size", config.partn.eigen_step_size)
        artn.set("nsmooth", config.partn.nsmooth)
        artn.set("neigen", config.partn.neigen)
        artn.set("alpha_mix_cr", config.partn.alpha_mix_cr)
        artn.set("nnewchance", config.partn.nnewchance)

        # Perpendicular relaxation
        if config.partn.nperp is not None:
            artn.set("nperp", config.partn.nperp)
        if config.partn.nperp_limitation is not None:
            artn.set("nperp_limitation", np.array(config.partn.nperp_limitation))
        else:
            artn.set("lnperp_limitation", False)

        # Convergence
        artn.set("forc_thr", config.partn.forc_thr)

        # Final push
        artn.set("push_over", config.partn.push_over)

        # RUN
        engine.command(f"minimize 1e-6 1e-8 10000 {config.partn.evalf_max}")
    except Exception as exc:
        recovery_error = None
        try:
            _reset_engine_state(engine, config, positions, cell, types)
        except Exception as recovery_exc:
            recovery_error = recovery_exc

        details = str(exc)
        if recovery_error is not None:
            details = (
                f"{details}; recovery failed with "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_SEARCH_RUNTIME_ERROR,
                message="Runtime error during event search.",
                details=details,
                variables={"central_atom_index": central_atom_idx},
            )
        )

    engine.command("unfix 10")
    _remove_frozen_fix(engine, "f_frozen_post", atoms_frozen)
    _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
    _delete_frozen_group(engine, atoms_frozen)

    # EXTRACT DATA
    if engine.rank == 0:
        if config.control.otfml:
            extrapolation_error = _build_extrapolation_error(
                get_otf_flags(engine),
                phase="search",
                message="Search extrapolated and must be retried.",
                variables={
                    "central_atom_index": central_atom_idx,
                },
            )
            if extrapolation_error is not None:
                return extrapolation_error

        err = artn.get_error()
        if err[0] == 0:
            # Results
            E_sad = artn.extract("etot_sad")
            E_min1 = artn.extract("etot_min1")
            E_min2 = artn.extract("etot_min2")

            if config.control.active_volume == True:
                min1positions, min2positions, saddlepositions, index_move = (
                    position_results_AV(config, artn, atom_map, positions)
                )
            else:
                min1positions = artn.extract("tau_min1")
                min2positions = artn.extract("tau_min2")
                saddlepositions = artn.extract("tau_sad")

                # find atom that moves the most
                dist = (min1positions - saddlepositions) ** 2
                dist = dist.sum(axis=-1)
                dist = np.sqrt(dist)
                dist[dist > config.atomicenvironment.rcut] = (
                    0  # if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
                )
                index_move = np.argmax(dist)

            delr_threshold = config.eventsearch.delr_thr
            delr1 = compute_delr_l2(positions, min1positions, cell=cell)
            delr2 = compute_delr_l2(positions, min2positions, cell=cell)
            npart1 = count_moved_atoms(
                positions, min1positions, delr_threshold, cell=cell
            )
            npart2 = count_moved_atoms(
                positions, min2positions, delr_threshold, cell=cell
            )
            npart12 = count_moved_atoms(
                min1positions, min2positions, delr_threshold, cell=cell
            )

            error_variables = {
                "delr1": delr1,
                "delr2": delr2,
                "npart1": npart1,
                "npart2": npart2,
                "npart12": npart12,
            }

            if npart12 == 0:
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_MINIMA_NOT_DISTINCT,
                        message=(
                            "min1 and min2 are identical within delr threshold {}"
                        ).format(delr_threshold),
                        variables=error_variables,
                    )
                )

            if npart1 != 0 and npart2 != 0:
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_MINIMA_NOT_MATCH_POSITIONS,
                        message=(
                            f"Neither minimum matches the initial configuration within npart {min(npart1, npart2)}"
                        ),
                        variables=error_variables,
                    )
                )

            dE_forward = E_sad - E_min1
            dE_backward = E_sad - E_min2

            if delr1 <= delr2:
                return Ok(
                    EventSearchOutput(
                        central_atom_index=central_atom_idx,
                        dE_forward=dE_forward,
                        dE_backward=dE_backward,
                        min1_positions=min1positions,
                        saddle_positions=saddlepositions,
                        min2_positions=min2positions,
                        move_atom_index=index_move,
                    )
                )

            return Ok(
                EventSearchOutput(
                    central_atom_index=central_atom_idx,
                    dE_forward=dE_backward,
                    dE_backward=dE_forward,
                    min1_positions=min2positions,
                    saddle_positions=saddlepositions,
                    min2_positions=min1positions,
                    move_atom_index=index_move,
                )
            )
        else:
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND,
                    message="No event found",
                    details=err,
                )
            )


@capture_output()
def partn_refine(
    engine,
    config,
    central_atom_idx: int,
    positions=None,
    cell=None,
    types=None,
    saddle_idx=None,
    saddle_positions=None,
    minimize_outter_atoms: bool = True,
    num_reference_event: int | None = None,
    symmetry_index: int | None = None,
):
    try:
        # Set positions
        if config.control.active_volume == True:
            E_init, atom_map, central_lammps_id = partn_refine_AV(
                engine,
                config,
                central_atom_idx,
                positions,
                cell,
                types,
                saddle_idx,
                saddle_positions,
            )
        else:
            central_lammps_id = [central_atom_idx + 1]
            E_init = 0
            atom_map = None
            if positions is not None:
                set_positions(engine=engine, positions=positions)
                # small minimization with fix core atoms around central atom
                if minimize_outter_atoms:
                    minimize_freeze_core(
                        engine,
                        positions[central_atom_idx],
                        config.atomicenvironment.rcut,
                        maxiter=10,
                    )

        # INITILIZE ARTN
        artn = pypARTn.artn(engine="lammps")

        # LAMMPS COMMANDS
        engine.command(f"plugin load {artn.lib._name}")

        # SETUP ARTN
        artn.reset_input()
        # Control
        artn.set("filout", "artn.out." + str(engine.engine_id))
        artn.set("engine_units", "lammps/metal")
        artn.set("verbose", config.partn.verbosity)
        artn.set("struc_format_out", "none")
        artn.set("delr_thr", config.partn.delr_thr)

        # Exploration
        artn.set("lpush_final", False)
        artn.set(
            "lmove_nextmin", False
        )  # if true fortran runtime error when event not found
        artn.set("zseed", config.partn.zseed)

        # Initial push : Should not happen when refining
        artn.set("push_mode", config.partn.r_push_mode)
        if config.partn.push_mode == "rad":
            artn.set("push_dist_thr", config.partn.r_push_dist_thr)
        artn.set("push_step_size", config.partn.r_push_step_size)
        artn.set("push_ids", central_lammps_id)  # fortran start at 1
        artn.set("ninit", config.partn.r_ninit)

        # Lanczos
        artn.set("lanczos_min_size", config.partn.r_lanczos_min_size)
        artn.set("lanczos_max_size", config.partn.r_lanczos_max_size)
        artn.set("lanczos_disp", config.partn.r_lanczos_disp)
        artn.set("lanczos_eval_conv_thr", config.partn.r_lanczos_eval_conv_thr)

        # Eigenvector push
        artn.set("eigval_thr", config.partn.r_eigval_thr)
        artn.set("eigen_step_size", config.partn.r_eigen_step_size)
        artn.set("nsmooth", config.partn.r_nsmooth)
        artn.set("neigen", config.partn.r_neigen)
        artn.set("alpha_mix_cr", config.partn.r_alpha_mix_cr)
        artn.set("nnewchance", config.partn.r_nnewchance)

        # Perpendicular relaxation
        if config.partn.r_nperp is not None:
            artn.set("nperp", config.partn.r_nperp)
        if config.partn.r_nperp_limitation is not None:
            artn.set("nperp_limitation", np.array(config.partn.r_nperp_limitation))
        else:
            artn.set("lnperp_limitation", False)

        # Convergence
        artn.set("forc_thr", config.partn.r_forc_thr)

        # MAX attempt based on delr_sad (from initial position)
        # Fix that sometime, we go back to the minimum, so saddle point found is the minimum
        # When using a different seed it solves the problem

        max_attempts = config.partn.r_max_attempts
        inner_attempt = 0
        atoms_frozen = _make_frozen_group(engine, config, positions, types)
        _apply_frozen_fix(engine, "f_frozen_pre", atoms_frozen)

        while inner_attempt < max_attempts:
            exit_flag = False
            result = None  # for rank > 0
            engine.command("fix 10 all artn dmax {}".format(config.partn.r_dmax))
            _apply_frozen_fix(engine, "f_frozen_post", atoms_frozen)
            engine.command("min_style fire")
            # RUN
            engine.command(f"minimize 1e-6 1e-8 10000 {config.partn.r_evalf_max}")
            engine.command("unfix 10")
            _remove_frozen_fix(engine, "f_frozen_post", atoms_frozen)

            # EXTRACT DATA
            if engine.rank == 0:
                if config.control.otfml:
                    extrapolation_error = _build_extrapolation_error(
                        get_otf_flags(engine),
                        phase="refine",
                        message="Refinement extrapolated and must be retried.",
                        variables={
                            "central_atom_index": central_atom_idx,
                            "num_reference_event": num_reference_event,
                            "symmetry_index": symmetry_index,
                        },
                    )
                    if extrapolation_error is not None:
                        exit_flag = True
                        result = extrapolation_error
                if not exit_flag:
                    err = artn.get_error()
                    if err[0] == 0:  # No error
                        # Check if went back to minimum
                        delr_sad = artn.extract("delr_sad")
                        if delr_sad < config.partn.r_delr_sad_thr:  # Success
                            E_sad = artn.extract("etot_sad")
                            E_result = (
                                E_sad - E_init
                            )  # If AV's are on, will return the activation energy of the event. If not, jsut saddle energy
                            saddlepositions = artn.extract("tau_sad")

                            if config.control.active_volume == True:
                                saddlepositions_results = positions.copy()
                                for i, atom_idx in enumerate(atom_map):
                                    saddlepositions_results[atom_idx][0] = (
                                        saddlepositions[i][0]
                                    )
                                    saddlepositions_results[atom_idx][1] = (
                                        saddlepositions[i][1]
                                    )
                                    saddlepositions_results[atom_idx][2] = (
                                        saddlepositions[i][2]
                                    )
                            else:
                                saddlepositions_results = saddlepositions

                            exit_flag = True
                            result = Ok(
                                EventRefinementOutput(
                                    central_atom_index=central_atom_idx,
                                    saddle_positions=saddlepositions_results,
                                    E_saddle=E_result,
                                    num_reference_event=num_reference_event,
                                    symmetry_index=symmetry_index,
                                    refined="T",
                                )
                            )
            # Synchronize all ranks
            exit_flag = engine.local_engine_comm.bcast(exit_flag, root=0)
            if exit_flag:
                _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
                _delete_frozen_group(engine, atoms_frozen)
                return result

            inner_attempt += 1
            artn.set("zseed", config.partn.zseed)

        else:  # fail after max attemps
            _remove_frozen_fix(engine, "f_frozen_pre", atoms_frozen)
            _delete_frozen_group(engine, atoms_frozen)
            if engine.rank == 0:
                err = artn.get_error()
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_NOT_FOUND,
                        message="no event found",
                        details=err,
                    )
                )
            return None
    except Exception as exc:
        recovery_error = None
        try:
            _reset_engine_state(engine, config, positions, cell, types)
        except Exception as recovery_exc:
            recovery_error = recovery_exc

        details = str(exc)
        if recovery_error is not None:
            details = (
                f"{details}; recovery failed with "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        return Err(
            ErrorInfo(
                type=ErrorType.EVENT_REFINEMENT_RUNTIME_ERROR,
                message="Runtime error during event refinement.",
                details=details,
                variables={
                    "central_atom_index": central_atom_idx,
                    "num_reference_event": num_reference_event,
                    "symmetry_index": symmetry_index,
                },
            )
        )
