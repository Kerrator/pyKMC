# Active Volumes

Active Volume (AV) settings are defined in the `[ActiveVolume]` section of the input file,
which must be present when AV mode is enabled in the `[Control]` section.
The two main parameters are `ract` and `rmov` for the total radius of the AV and the
movable core radius, respectively (both in Angstrom, with defaults of 6.0 and 4.0 —
size them for your system, see the constraints below).

Example:

```INI
[Control]
...
active_volume = True
...

[ActiveVolume]
ract = 21
rmov = 15
```

**Constraint:** `ract` must be greater than or equal to `rmov`; `ract - rmov` is the frozen buffer thickness.

**Current limitation:** Active Volume mode supports single-element systems
only. The temporary LAMMPS box used for each Active Volume operation is
re-created with a single atom type, so do not enable it for alloys until the
Active Volume reset path creates and assigns all species and masses.

---

## General Idea

In very large systems with many defects, it can become memory intensive and slow to run event searches on the entire system. 
Active Volumes define a region around the selected search or refinement center where the operation is performed without needing to use the entire system.
The AV has two parameters, the active radius `ract` and the movable radius `rmov`: `ract` includes every atom copied into the temporary Active Volume, while atoms within `rmov` may move during event searches and refinement. Atoms between `rmov` and `ract` are frozen to provide accurate boundary forces.

<!-- TODO: add a figure illustrating the active and movable radii and the buffer region. -->

## Algorithm
1. **Reset LAMMPS Instance**: LAMMPS is cleared to prevent carryover of previous AV conditions
2. **AV Definition**: Find which atoms are to be included in AV, and which are movable
3. **Atom Map**: Create map between full system and AV system atom indices, types and positions
4. **AV Creation**: Atoms are created within AV

This then branches depending on whether the operation is an event search or a refinement.

**Event Search**: Event search continues the same as without Active Volumes. Once results are obtained, the positions are mapped 
back to those of the full system.

**Refinement**: Unlike a non-AV refinement, the positions sent for refinement are not at the saddle point. First, the 
system in its current state is sent, the AV defined, then the potential energy `E_init` is calculated. This is needed as
regular refinements compare the saddle point energy to that of the full system, however in the AV the saddle point 
energy is in reference to that within the AV, not the full system. The event being refined is then applied to the atoms
saved in the local atomic environment, then the system is refined. The refined activation energy is then calculated from
`E_saddle - E_init` and the positions mapped back to the full system. 


`ract` must be strictly greater than `[AtomicEnvironment].rcut`, or event search
raises an error. There is no equivalent automatic check on `rmov`, so also choose
`ract >= rmov` and make `rmov` large enough to contain the reconstructed local
environment. For reliable refinement, size the frozen buffer `ract - rmov` to be
at least as thick as the interatomic potential's interaction cutoff.

A debug mode to check if the AV is large enough can be toggled in `[ActiveVolume]` by setting `AV_debug = True`. This
will minimize the AV during refinement before the event is applied, and print the energy before and after together
with their **percentage** difference. The system being sent for refinement is already minimized, so the difference
should be essentially zero; pyKMC does not enforce an automatic pass/fail threshold, so inspect the printed values.
A noticeable difference means the settings for the AV need to be adjusted — typically the buffer region is thinner
than the cutoff of the potential being used, so movable atoms next to the buffer experience improper forces. 

