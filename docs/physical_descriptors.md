# Physical descriptors and constraint transport

`pykmc.physics` defines immutable descriptors shared by engine preflight, the
prefactor service, requests, and event-table comparison interfaces. An engine
snapshot preserves the complete potential species order and actual masses after
potential initialization, including species with no atoms in a crop. A request
must agree with that snapshot before a scratch engine is created.

Active-volume search and refinement use the initialized full-system species
order and masses when rebuilding their shared native engine. They retain every
potential type slot, including absent species, and reapply the authoritative
masses after potential commands that may overwrite them. The remembered full
descriptor remains available for restoration. Standalone mapping helpers with
no descriptor retain the default alphabetical species/ASE mass convention.

The native scatter boundary normalizes periodic images in the source cell
before converting to LAMMPS coordinates. Different atoms may carry different
integer image translations. Nonperiodic lattice directions retain their
physical displacement; normalization neither reorders atoms nor modifies the
input array. The AV refinement helper uses this same engine boundary.

A failed full-system restore retains its original descriptor and pending state
even if the native error wrapper closes the handle. Retrying starts a fresh
instance when necessary. Restoration verifies the force snapshot and actual
replayed species/masses before reporting success; changed physics remains an
explicit failure. Closing an intact engine intentionally does not restart it.

The force-model snapshot identifies the command definition and coefficient-file
contents. Moving an identical file does not change its content identity. Editing
a file at the same path does. The supported fingerprint parsers cover single-file
SW, Tersoff, EAM, EAM/alloy and EAM/fs definitions and numeric LJ/cut and zero
potentials. Other command forms remain explicitly non-reusable because their
complete dependencies are unknown. Dynamic commands are also non-reusable. Fresh
calculations can retain this limitation; serialization cannot remove it.

The physical descriptor includes Hessian region/step/tolerance choices and the
user constraint policy. Preminimization adds the engine's `min_style` and
`frz_min` inputs only while enabled. Temperature, fallback `k0`, and prefactor
acceptance windows are separate rate/acceptance policies and do not change a
Vineyard calculation's physical identity. Changing those policies still requires
consumers to recompute rates or revalidate acceptance.

`ResolvedConstraints` retains full source identities, current local-to-global
correspondence, frozen global identities and their immutable reference positions.
Crop operations explicitly remap local rows without reinterpreting global
indices or losing fixed coordinates. This endpoint restriction is distinct from
the vibrational free set. Transporting it does not itself apply constraints during
minimization or remove degrees of freedom from a Hessian.

`HTSTEventRequest.calculation_identity` snapshots all three full geometries,
atom correspondence, cell, periodic axes, center, direction, constraints and the
supplied common free set. Build it on the full request before cropping. A missing
free set or descriptor remains explicit. Batch `event_key` values and dataframe
labels do not identify scientific calculations.

`ReferenceEventTable.current_descriptor` is current service context;
`compare_physics` reports compatible, incompatible or unknown producing physics.
These comparison interfaces do not independently enforce cache invalidation or
prove whole-event equivalence. A stored estimate must keep its own producing
provenance; attaching a service or saving a table must never establish missing
provenance for that estimate. Geometry and constraint consumption, persistence
policy, and whole-event matching remain separate consumers of this contract.

## Reference estimate persistence

HTST/RPA reference pickles use schema 2. Their metadata contains a descriptor
registry, immutable directional calculation records, and explicit associations
from logical reference IDs to those records. Each calculation retains the full
source triplet before preminimization and the full produced triplet before any
Hessian crop, together with actual free/crop atom identities and method. A digest
also binds the associated local event geometry and barrier; changing dataframe
labels does not change the association.

Saving only serializes these facts and clearly labeled rate-policy context.
Attaching a different service cannot change an old producer. Unknown or
superseded values remain in diagnostic history, outside selectable `nu0` columns.
Schema-1 tables and rows without a producing calculation keep their event
geometry and use an explicit unavailable status with the current `k0` fallback.
Repeated saving and loading cannot supply their missing provenance.

Loading a schema-1 table (or any HTST pickle without per-calculation
provenance) is loud: exactly one `WARNING` names the file, the stored and
current schema versions, the number of rows kept and the number of accepted
estimates demoted to `legacy`/`k0`, and the recovery path. Reference prefactors
cannot be rebuilt from the stored local crops, so recovery means running with a
prefactor service (refined sites then get site estimates, `nu0_source = site`)
or regenerating the catalogue under the current schema; selection never
launches a recomputation for such rows. Their original frequencies stay in the
archive history for inspection.

Loading, reference-estimate retrieval, and the catalogue subset passed to
refinement validate current physics before inheritance. Verified same-input
calculations reuse their frequencies. Current inclusive frequency windows are
applied separately, and temperature/`k0` changes rebuild rates without Hessians.
Complete source snapshots can be recalculated after incompatible physics or
numerical changes; local event crops alone cannot reconstruct missing atoms.
Recalculation resolves current initialized user constraints and AV policy while
preserving the saved AV search center separately from the HTST moving atom.
Triplets that would move a newly fixed atom remain unavailable.

When initialized user constraints and the AV policy are unchanged, reload keeps
the actual source-resolved restriction, including its original pre-search center.
Relaxing minimum 1 does not move that center or redefine its fixed membership.
Changes to masses or numerical settings still require current calculations.

Restart recomputation requests all three full-system potential energies after
preminimization and before cropping. It updates the directional barrier as well
as the frequency; ordinary prefactor calls incur no extra energy evaluations.
The recomputed barrier is adopted only when the kernel accepts the geometry, or
rejects it for a reason that leaves the stationary triplet intact (frequency
window, empty free region, non-finite prefactor). A rejection that invalidates
the geometry (`nonstationary_geometry`, `unstable_minimum`,
`saddle_not_first_order`, `mode_count_mismatch`, `nonfinite_hessian`) keeps the
catalogued barrier, records the attempt with both values in the archive history
and logs a warning; the row uses the `k0` fallback on the catalogued barrier.
The superseded calculation remains in the registry and history. Fresh opaque
force-model results can be used in their producing context, but serialization
does not make them reusable: a subsequent load needs a new calculation or an
explicit fallback. Worker and transport failures still propagate.

## Active site estimates and recycling

An unchanged event center does not establish unchanged local curvature.
HTST/RPA active rows retain the full submitted source snapshot, initialized
force-model and mass descriptor, actual numerical settings and restrictions,
and stable atom correspondence. A site value also retains its actual immutable
forward calculation. Its association covers the stored saddle and final crops,
barrier, logical reference, status and frequency; replacing a row cannot silently
transfer that calculation.

Recycling validates these dependencies before refinement skips and before rate
selection. Any full-source geometry change conservatively invalidates the row,
including movement outside its stored crop. Changed masses, force-model
identity, constraints, numerical settings or crop membership also invalidate it.
The ordinary dispatcher can then rebuild a current full saddle and request a
new site calculation. When full geometry is unavailable, an explicit crop-ID
handoff can retain a labeled reference approximation or `k0` fallback; it cannot
manufacture a full stationary saddle by overlaying the crop.

Pure source or crop reordering preserves stable global IDs and needs no new
Hessian. Temperature changes update rates using the current rate facade while
retaining an otherwise valid frequency. The current inclusive frequency window
still applies; widening a window retries a previously rejected site calculation,
including one whose row temporarily used a valid reference approximation.

Fresh nonreusable force-model results and explicit fallback contexts can be used
by their producing service in the current step, subject to the same exact source
and row checks. This transient allowance ends at pruning or a service change.
An opaque descriptor remains nonreusable and cannot establish recycled validity.
Coordinate representations differing by a periodic image may conservatively
trigger rebuilding, while the displacement/distance filter itself respects the
source's actual periodic axes.

KMC restart initialization evaluates energy at the saved System coordinates
without relaxing them. The Python configuration and native energy therefore
refer to the same geometry when refinement begins. Fresh runs retain their
initial minimization and adopt its returned coordinates and energy.

## Directional event identity

HTST/RPA admission starts with two reciprocal directional rows. Equal topology,
saddle crops, barriers or prefactors alone do not establish self-reversal. A
merge needs one species- and mass-preserving rigid transformation/permutation
that maps the entire minimum/saddle/minimum triplet, including the full source
before preprocessing and the full produced geometry before Hessian cropping.
It must preserve actual free/crop sets, user and execution constraints, fixed
reference positions, and the source's periodic lattice. Local grey coloring
cannot waive these physical checks.

The existing 0.01 eV barrier comparison and 5% accepted-frequency agreement are
necessary conditions; the frequency tolerance remains uncalibrated. Rejected
or unavailable estimates cannot prove a merge. The map search is conservative:
a failed witness retains directions and does not prove mathematical
inequivalence. Opaque force definitions and incomplete source provenance cannot
authorize collapse. Reflections are supported only within the fingerprinted,
parity-invariant pair-potential scope.

A known forward duplicate can avoid a new worker request only after read-only
validation of its current accepted producer and one full source map. Otherwise
actual results are retained before final identity comparison, including events
that arrived together in one batch. Proven merges redirect incoming aliases,
retain both immutable producing records and the removed row's history, and
allocate future logical IDs above surviving and archived IDs. Returned admitted
frames contain only surviving resolved rows. Constant-mode admission keeps its
original topology/saddle policy.

Discarding both rows into an older pair also requires a coherent reciprocal
pair with one canonical producing context, or a self-linked survivor after
directly proving the new pair self-reverse. Separate approximate comparisons
to one old estimate cannot replace the direct directional agreement test.

## Source-resolved endpoint constraints

`ResolvedConstraints` can carry the source cell/PBC, event center identity and
coordinate, and movable radius alongside the immutable fixed references. Resolve
the union of user constraints and outside-radius atoms before changing any source
coordinates. Crop copies retain source context and map frozen global identities to
local rows. Active-volume membership uses the source periodic axes.

Search, refinement and reconstruction resolve that same union from the full
source before cropping. The initialized user mask is carried separately: moving
an unfrozen atom across a configured spatial boundary does not silently change
which source identities were frozen. Returned search vertices retain their
original full-coordinate frame for prefactor evaluation, even when the event
catalogue stores a recentered representation. Basin state copies preserve these
source identities.

During constrained pARTn searches and refinements, force masking is followed by
a temporary LAMMPS `fix external` callback that also zeros fixed-atom velocities.
ARTn's perpendicular relaxation can otherwise rewrite velocities and allow FIRE
to move fixed atoms despite zero forces. This requires the LAMMPS Python callback
API. The callback uses current local atom tags after redistribution, leaves free and
ghost velocities unchanged, and never projects returned coordinates. Worker
failures are agreed across the engine communicator before native work resumes;
only operation-owned callbacks and fixes are removed. Callback failures remain
recorded until the existing finite native minimization limit returns, then reject
the result and restore the source. This can take the configured evaluation budget;
it does not expire the native timer and poison later minimizations. A failed
cleanup remains an explicit pending-restoration error.

Each operation explicitly destroys its owned pARTn wrapper after copying output
arrays, including when an exception is retained by the caller. Relying on Python
garbage collection can leave native Lanczos state alive and corrupt a later
refinement. Full-system failures also remove their temporary pARTn/frozen groups
and restore native entry coordinates; incompatible returned fixed coordinates
are checked inside that transaction. Active-volume calls retain full source
replay on every exit path.

A reconstruction validates its claimed minimum/saddle/minimum against those
references before protecting working pushes. Both endpoint dispatches receive the
same payload and actual species labels. An explicit payload passed to
`minimize_with_results` selects transactional behavior: it returns a detached
endpoint and energy, removes only its own temporary native group/fix, and restores
entry coordinates and computes. Ordinary initialization minimization retains its
updating behavior. The live Python source is not changed during reconstruction.

A secondary cleanup failure preserves the initiating exception and leaves a
pending-restoration marker. A closed native handle cannot retain arbitrary user
fixes: that path reports failure rather than claiming a complete resource replay.
Endpoint relaxation restrictions remain distinct from the common vibrational
free-coordinate selection.

## Periodic axes during reconstruction

`System` normalizes a boolean PBC scalar to three axes and copies boolean
three-axis inputs. Invalid shapes or nonboolean flags reject before position
updates. An unspecified empty system starts nonperiodic. Position updates,
reference-event recentering, neighbor membership, PSR unwrapping, refinement
geometry and both reconstruction pushes/comparisons use the source axes.
Negative coordinates on a nonperiodic axis are physical coordinates and remain
unchanged by wrapping. A whole-box displacement on such an axis is not an
equivalent image. Neighbor lists continue to require orthorhombic cells.

This changes mixed/open-boundary geometry and neighbor membership from the old
all-periodic assumption. Fully periodic image equivalence remains supported.
It does not establish basin acceleration or constrained HTST acceptance by
itself; those require their own numerical and integration checks.

## Constrained HTST calculations

The request carries the initialized user constraint snapshot separately from
its execution mask. The user snapshot records the policy that resolved it;
a different or unknown policy cannot establish authority for a descriptor.
The execution mask may add AV locks, but must preserve all user-fixed source
identities and reference coordinates, including atoms outside a crop. Service
requests remap initialized constraints by source identity when rows are
permuted. A stateless full request can declare minimum 1 as its original source;
resolve that policy before relaxation or cropping and retain the snapshot.

All three event geometries must obey the immutable fixed coordinates, allowing
only images along the actual periodic axes. Preminimization holds the union of
the event core and fixed atoms. Every Hessian instead uses one common core
selection with fixed atoms removed, including for explicit free selections.
An empty common set returns an unavailable estimate. Fixed atoms are never
finite-displaced by the Hessian calculation.

The native event adapter checks raw forces after preminimization and cropping,
before the undisplaced saddle and each requested minimum reaches the Hessian.
The largest Euclidean force norm on a common free atom must be at most
`[rateconstant] force_tol`, a finite positive value in eV/Angstrom (default
0.005). Fixed atoms may carry reaction forces. Above-limit or nonfinite free
forces return `nonstationary_geometry` for affected directions; the saddle must
also have exactly one unstable common-subspace mode. This is a finite force
convergence criterion, recorded in the producing descriptor. It does not imply
exact zero force. Low-level derivative helpers and the generic Hessian-only
kernel cannot check stationarity; their callers must establish it separately.
