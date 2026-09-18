"""Per-event orchestration: two directional Vineyard estimates sharing one saddle.

The orchestrator validates the request, selects one common free-atom subset,
obtains and classifies the saddle Hessian exactly once, then evaluates each
direction independently. Rejection precedence is "saddle first": when the shared
saddle spectrum is not a first-order saddle both directions are rejected with
``SADDLE_NOT_FIRST_ORDER`` and no minimum Hessian is requested. Scientific
rejections (:class:`PrefactorRejected`) become ``status="rejected"`` on the
affected direction only; every other exception propagates to the caller,
because a broken payload, a dead engine or a programming error must not
masquerade as a physical fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .free_region import select_free_indices
from .hessian import HessianFn
from .normal_modes import ModeSpectrum, normal_modes_from_hessian
from .request import HTSTEventRequest, HTSTRequestError
from .result import (
    DirectionalPrefactor,
    EventPrefactors,
    PrefactorRejected,
    PrefactorRejection,
)
from .vineyard import vineyard_from_spectra

# Frozen-boundary partial Hessians carry no translational invariance, so nothing is
# projected out. The analysis-package default of 3 is never used in production.
N_ZERO_MODES_PARTIAL_HESSIAN: int = 0


def _hessian_for(
    hessian_fn: HessianFn, positions: np.ndarray, free: np.ndarray, label: str
) -> np.ndarray:
    """Call ``hessian_fn`` and check that it returned a real ``(3F, 3F)`` array."""
    raw = hessian_fn(positions, free)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"hessian_fn returned a complex array for the {label} geometry; "
            "the mass-weighted Hessian must be real"
        )
    h_mw = np.asarray(raw, dtype=float)
    expected = (3 * free.size, 3 * free.size)
    if h_mw.shape != expected:
        raise ValueError(
            f"hessian_fn returned shape {h_mw.shape} for the {label} geometry, "
            f"expected {expected} for {free.size} free atoms"
        )
    return h_mw


def _reason(rej: PrefactorRejected) -> str:
    """Return a non-empty human-readable reason for ``rej``.

    A ``hessian_fn`` may raise :class:`PrefactorRejected` with an empty detail;
    the rejection code then doubles as the reason so the typed result stays valid.
    """
    return rej.detail or rej.reason_code.value


def _direction(
    hessian_fn: HessianFn,
    positions: np.ndarray,
    free: np.ndarray,
    sad_spec: ModeSpectrum,
    n_negative_saddle: int,
    request: HTSTEventRequest,
    label: str,
) -> DirectionalPrefactor:
    """Evaluate one direction against the shared, already classified saddle."""
    n_free = int(free.size)
    n_positive_min: int | None = None
    try:
        h_min = _hessian_for(hessian_fn, positions, free, label)
        min_spec = normal_modes_from_hessian(
            h_min,
            expect_saddle=False,
            zero_mode_tol=request.settings.zero_mode_tol,
            n_zero_modes=N_ZERO_MODES_PARTIAL_HESSIAN,
        )
        n_positive_min = min_spec.n_positive
        nu0_hz = vineyard_from_spectra(min_spec, sad_spec)
    except PrefactorRejected as rej:
        return DirectionalPrefactor.rejected(
            rej.reason_code,
            _reason(rej),
            n_free=n_free,
            n_positive_min=n_positive_min,
            n_negative_saddle=n_negative_saddle,
        )
    lo = request.settings.nu0_min_hz
    hi = request.settings.nu0_max_hz
    if nu0_hz < lo or nu0_hz > hi:
        return DirectionalPrefactor.rejected(
            PrefactorRejection.OUT_OF_WINDOW,
            f"nu0 = {nu0_hz:.6e} Hz outside the inclusive window [{lo:.6e}, {hi:.6e}] Hz",
            n_free=n_free,
            n_positive_min=n_positive_min,
            n_negative_saddle=n_negative_saddle,
        )
    return DirectionalPrefactor.accepted(
        nu0_hz,
        n_free=n_free,
        n_positive_min=n_positive_min,
        n_negative_saddle=n_negative_saddle,
    )


def compute_event_prefactors(
    request: HTSTEventRequest,
    hessian_fn: HessianFn,
    *,
    method: str = "fd",
    free_indices: Any | None = None,
) -> EventPrefactors:
    """Compute the forward and backward Vineyard prefactors of one event.

    Parameters
    ----------
    request : HTSTEventRequest
        Validated on entry; violations raise.
    hessian_fn : Callable
        ``hessian_fn(positions, free_indices) -> H_mw`` returning the real
        ``(3F, 3F)`` mass-weighted Hessian in eV / (amu Å²) for the free atoms in
        ``free_indices`` order, symmetric to within
        :data:`pykmc.htst.normal_modes.SYMMETRY_ATOL` (1e-8 absolute). Adapters
        assembling it from raw engine output must return ``0.5 * (H + H.T)``;
        asymmetry beyond the tolerance, a complex dtype or a wrong shape is a
        plumbing error and raises ``ValueError``. It may raise
        :class:`PrefactorRejected` (any code, detail optional) to reject the
        direction(s) that need that geometry. Called exactly once for the saddle,
        then once per minimum only when the saddle is a first-order saddle.
    method : str, optional
        Label recorded on the result, e.g. ``"fd"`` or ``"lammps_eskm"``.
    free_indices : array_like, optional
        Caller-supplied free selection (global indices) overriding the sphere of
        radius ``settings.free_radius`` around ``center_index`` in the ``min1``
        geometry. One selection feeds all three Hessians, so the backward
        direction's sphere is centred on the moving atom's ``min1`` position, not
        its ``min2`` position. An empty selection rejects both directions with
        ``EMPTY_FREE_REGION``.

    Returns
    -------
    EventPrefactors
        Per-direction results; ``forward`` is ``min1 -> saddle`` and ``backward`` is
        ``min2 -> saddle``.

    Raises
    ------
    HTSTRequestError
        If the request fails validation (including ``HTSTGeometryError``).
    ValueError
        If ``method`` is empty, ``free_indices`` is malformed, ``hessian_fn`` is
        not callable, or ``hessian_fn`` returns a wrong shape, a complex array or
        an asymmetric matrix.
    Exception
        Anything raised by ``hessian_fn`` other than :class:`PrefactorRejected`.

    """
    if not isinstance(request, HTSTEventRequest):
        raise HTSTRequestError(
            f"request must be an HTSTEventRequest, got {type(request).__name__}"
        )
    request.validate()
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty str")
    if not callable(hessian_fn):
        raise ValueError("hessian_fn must be callable")

    min1 = np.asarray(request.min1_positions, dtype=float)
    saddle = np.asarray(request.saddle_positions, dtype=float)
    min2 = np.asarray(request.min2_positions, dtype=float)
    n_atoms = min1.shape[0]
    center = int(request.center_index)

    if free_indices is None:
        free = select_free_indices(
            min1, center, request.settings.free_radius, request.cell, request.pbc
        )
    else:
        free = np.asarray(free_indices)
        if free.ndim != 1:
            raise ValueError(f"free_indices must be one-dimensional, got {free.shape}")
        if free.size and not np.issubdtype(free.dtype, np.integer):
            raise ValueError(f"free_indices must be integers, got dtype {free.dtype}")
        free = np.sort(free.astype(int))
        if free.size and (free[0] < 0 or free[-1] >= n_atoms):
            raise ValueError(f"free_indices out of range for {n_atoms} atoms: {free}")
        if free.size and len(np.unique(free)) != free.size:
            raise ValueError(f"free_indices contains duplicates: {free}")

    n_free = int(free.size)
    if n_free == 0:
        detail = (
            f"no free atoms selected for event {request.event_key!r} "
            f"(center_index={center}, free_radius={request.settings.free_radius})"
        )
        empty = DirectionalPrefactor.rejected(
            PrefactorRejection.EMPTY_FREE_REGION,
            detail,
            n_free=0,
            n_positive_min=None,
            n_negative_saddle=None,
        )
        return EventPrefactors(
            event_key=request.event_key,
            forward=empty,
            backward=empty,
            method=method,
            n_free=0,
            settings=request.settings,
        )

    # The saddle Hessian is obtained and classified exactly once and shared.
    try:
        h_sad = _hessian_for(hessian_fn, saddle, free, "saddle")
        sad_spec = normal_modes_from_hessian(
            h_sad,
            expect_saddle=True,
            zero_mode_tol=request.settings.zero_mode_tol,
            n_zero_modes=N_ZERO_MODES_PARTIAL_HESSIAN,
        )
    except PrefactorRejected as rej:
        # The saddle spectrum was never computed: hessian_fn itself raised (any
        # code) or the Hessian was non-finite. Both directions share the verdict.
        both = DirectionalPrefactor.rejected(
            rej.reason_code,
            _reason(rej),
            n_free=n_free,
            n_positive_min=None,
            n_negative_saddle=None,
        )
        return EventPrefactors(
            event_key=request.event_key,
            forward=both,
            backward=both,
            method=method,
            n_free=n_free,
            settings=request.settings,
        )

    n_negative_saddle = sad_spec.n_negative
    saddle_rejection = sad_spec.rejection()
    if saddle_rejection is not None:
        # Saddle first: a saddle that is not first order rejects both directions
        # and no minimum Hessian is requested (each one costs 6F force calls).
        both = DirectionalPrefactor.rejected(
            *saddle_rejection,
            n_free=n_free,
            n_positive_min=None,
            n_negative_saddle=n_negative_saddle,
        )
        return EventPrefactors(
            event_key=request.event_key,
            forward=both,
            backward=both,
            method=method,
            n_free=n_free,
            settings=request.settings,
        )

    forward = _direction(
        hessian_fn, min1, free, sad_spec, n_negative_saddle, request, "min1"
    )
    backward = _direction(
        hessian_fn, min2, free, sad_spec, n_negative_saddle, request, "min2"
    )
    return EventPrefactors(
        event_key=request.event_key,
        forward=forward,
        backward=backward,
        method=method,
        n_free=n_free,
        settings=request.settings,
    )


__all__ = ["N_ZERO_MODES_PARTIAL_HESSIAN", "compute_event_prefactors"]
