# Physical descriptors and constraint transport

`pykmc.physics` defines immutable descriptors shared by engine preflight, the
prefactor service, requests, and event-table comparison interfaces. An engine
snapshot preserves the complete potential species order and actual masses after
potential initialization, including species with no atoms in a crop. A request
must agree with that snapshot before a scratch engine is created.

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
