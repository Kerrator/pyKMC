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

## Source-resolved endpoint constraints

`ResolvedConstraints` can carry the source cell/PBC, event center identity and
coordinate, and movable radius alongside the immutable fixed references. Resolve
the union of user constraints and outside-radius atoms before changing any source
coordinates. Crop copies retain source context and map frozen global identities to
local rows. Active-volume membership uses the source periodic axes.

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
