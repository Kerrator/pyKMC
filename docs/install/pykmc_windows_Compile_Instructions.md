# pyKMC Windows Installation Instructions

**Tested on:** Windows 10 Home 22H2 (build 19045+) and Windows 11 22H2/23H2/24H2
**Strategy:** Run pyKMC inside **WSL2 (Windows Subsystem for Linux)** with Ubuntu 24.04 LTS.
**Python:** 3.9–3.13 (Ubuntu 24.04 ships 3.12)
**LAMMPS:** `stable_22Jul2025_update3`

> **Why WSL2 and not a native Windows port?** pyKMC's stack (LAMMPS + pARTn + IRA) is Fortran/MPI-heavy and assumes a POSIX toolchain. There is no Microsoft Fortran compiler, MS-MPI is not API-compatible with OpenMPI, and pARTn/IRA's Makefiles assume a Unix shell. Building all of this natively on Windows is multi-week yak-shaving; running it inside WSL2 is ~native Linux performance with the **existing tested Linux install script**. See `pykmc_linux_Compile_Instructions.md` for the Linux side.

> **Tip:** For an automated install, run [`install_pykmc_windows.ps1`](install_pykmc_windows.ps1) from an **elevated PowerShell** (Run as Administrator) — it does the Windows-side WSL2 + Ubuntu setup, then delegates the Linux-side bootstrap to [`setup_pykmc_distro.sh`](setup_pykmc_distro.sh) which in turn runs [`install_pykmc_linux.sh`](install_pykmc_linux.sh) inside the new distro. See **Automated install** below. To install manually, skip to [Section 0](#0-system-prerequisites).
>
> **Three files make up the Windows install** (all in `docs/install/`):
> - `pykmc_windows_Compile_Instructions.md` — this guide.
> - `install_pykmc_windows.ps1` — Windows-side driver (PowerShell 5.1 compatible). Enables WSL features, installs Ubuntu 24.04, relocates the VHD to a chosen drive.
> - `setup_pykmc_distro.sh` — Linux-side bootstrap, called by the PS1 inside WSL. Creates the Linux user, writes `/etc/wsl.conf`, then runs `install_pykmc_linux.sh`. Kept as a separate `.sh` because PowerShell 5.1 has parser quirks that make embedding multi-line bash unreliable.

---

## Automated install (recommended)

`install_pykmc_windows.ps1` does:

1. Verifies Windows version, virtualization, and free disk space.
2. Enables the `WSL` and `VirtualMachinePlatform` Windows features (requires admin + reboot on first run).
3. Sets WSL default version to 2 and updates the kernel.
4. Installs Ubuntu 24.04 LTS, then **relocates** its VHD to a chosen drive (default `P:\WSL\Ubuntu`) so the build tree lives on a fast NVMe SSD with headroom.
5. Inside the new Ubuntu, runs the project's existing `install_pykmc_linux.sh` to build LAMMPS, IRA, pARTn, and pyKMC.

The script is **resumable**: on first run it enables features and asks you to reboot; on the second run it picks up at distro install. Re-running after a successful install is a no-op.

### Run it

Open **PowerShell as Administrator** (right-click PowerShell → *Run as Administrator*), then:

```powershell
# Allow the script to run for this session only:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# From the folder containing the script:
.\install_pykmc_windows.ps1
```

To put the install on a different drive (default is `P:\WSL\Ubuntu`):

```powershell
.\install_pykmc_windows.ps1 -InstallPath 'D:\WSL\Ubuntu'
```

To save a log:

```powershell
.\install_pykmc_windows.ps1 *>&1 | Tee-Object install.log
```

When it finishes you'll have:

- `P:\WSL\Ubuntu\ext4.vhdx` — the Ubuntu virtual disk (~20 GB after install).
- A `pykmc` distro registered with WSL: `wsl -d pykmc` opens a shell.
- Inside the distro, `~/pykmc/` containing `pykmc_env/`, `lammps/`, `IterativeRotationsAssignments/`, `artn-plugin/`, and `activate.sh`.

---

## 0. System prerequisites

| Requirement | Why | How to check |
|---|---|---|
| Windows 10 build ≥ 19041 (2004) **or** Windows 11 | WSL2 requires it | `winver` |
| 64-bit OS, x64 CPU | WSL2 requirement | `systeminfo` |
| CPU virtualization enabled in BIOS/UEFI (VT-x / AMD-V) | Hyper-V backend for WSL2 | Task Manager → Performance → CPU → "Virtualization: Enabled" |
| ≥ 30 GB free on the chosen install drive | Ubuntu + LAMMPS source/build + venv | `Get-Volume` |
| Admin rights | To enable Windows features (Section 1) **and** to update the WSL kernel via `wsl --update` (Section 1, post-reboot). Section 2 onward does not require admin. | — |

If virtualization is disabled, enable it in your BIOS/UEFI (look for *Intel VT-x*, *AMD-V*, or *SVM Mode* under CPU/Advanced settings).

---

## 1. Enable WSL2 features (admin required, requires reboot)

In **PowerShell as Administrator**:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Reboot:

```powershell
Restart-Computer
```

After reboot, re-open **PowerShell as Administrator** (`wsl --update` installs a kernel update and needs elevation), then set WSL2 as the default version and update the kernel:

```powershell
wsl --set-default-version 2
wsl --update
```

If `wsl --update` reports `WSL is finishing an upgrade...`, wait a few seconds and re-run; the kernel update is finalizing in the background.

---

## 2. Install Ubuntu 24.04 LTS

Install Ubuntu without launching it (we'll relocate it before first boot):

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

Verify:

```powershell
wsl --list --verbose
```

Expected output (something like):

```
  NAME            STATE           VERSION
* Ubuntu-24.04    Stopped         2
```

---

## 3. Relocate the distro VHD to a fast drive

> **Why:** by default WSL puts the `ext4.vhdx` under `%LOCALAPPDATA%\Packages\...` on the system drive (C:). Relocating to a fast NVMe drive with headroom (e.g. P:) gives faster compile/I-O and keeps the OS drive uncluttered. **On Windows 11 you can use `wsl --install --location <path>` directly; on Windows 10 you must use the export/unregister/import recipe below.**

Pick a destination folder on a fast drive with ≥ 30 GB free. For this guide we use `P:\WSL\Ubuntu`.

```powershell
$Dst = 'P:\WSL\Ubuntu'
New-Item -ItemType Directory -Force -Path $Dst | Out-Null

# Export the freshly-installed distro to a tar:
wsl --export Ubuntu-24.04 "$Dst\Ubuntu-24.04.tar"

# Unregister the original (this deletes the VHD on C: but the tar above is safe):
wsl --unregister Ubuntu-24.04

# Re-import into the new location with a friendlier name:
wsl --import pykmc "$Dst" "$Dst\Ubuntu-24.04.tar" --version 2

# Optional: delete the tar once import succeeds
Remove-Item "$Dst\Ubuntu-24.04.tar"

wsl --list --verbose
```

You should now see a distro named `pykmc` in WSL list. Its VHD lives at `P:\WSL\Ubuntu\ext4.vhdx`.

### Set a default Linux user

A reimported distro defaults to `root`. Create a regular user and make it the default:

```powershell
wsl -d pykmc -u root -- bash -c "useradd -m -s /bin/bash -G sudo pykmc && passwd -d pykmc && echo 'pykmc ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/pykmc"
```

Then create `/etc/wsl.conf` so the next launch lands you in this user:

```powershell
wsl -d pykmc -u root -- bash -c "printf '[user]\ndefault=pykmc\n' >/etc/wsl.conf"
wsl --terminate pykmc
```

---

## 4. First-boot Ubuntu setup

```powershell
wsl -d pykmc
```

You're now inside Ubuntu as user `pykmc`. Update the package index and base system:

```bash
sudo apt update && sudo apt upgrade -y
```

This usually takes 2–5 minutes the first time.

---

## 5. Run the existing pyKMC Linux installer

Still inside the WSL Ubuntu shell, follow [`pykmc_linux_Compile_Instructions.md`](pykmc_linux_Compile_Instructions.md) — every step works unmodified. The fastest path is the automated script:

```bash
# From your home directory (NOT /mnt/c — see "Performance trap" below):
cd ~
mkdir -p pykmc-install && cd pykmc-install

# Download the install script directly:
curl -fsSLO https://raw.githubusercontent.com/hugomoison/pyKMC/develop/docs/install/install_pykmc_linux.sh
chmod +x install_pykmc_linux.sh

./install_pykmc_linux.sh 2>&1 | tee install.log
```

The script will prompt once for your sudo password (passwordless via the wsl.conf step above means it's instant), install build tools via `apt`, then compile LAMMPS for ~10–20 minutes.

When done, you'll have `~/pykmc-install/pykmc/` with `pykmc_env/`, `lammps/`, etc. — exactly the same layout as a native Linux install.

### Performance trap: don't put your work on `/mnt/c`

The `/mnt/c`, `/mnt/d`, `/mnt/p` mount points let WSL access your Windows drives, but file IO across the boundary is **10–20× slower** than the native WSL filesystem. Always keep your source tree, build dir, and simulation outputs inside the Linux home (`~/`), which lives in the relocated `ext4.vhdx`. You can still `code .` in WSL and edit from Windows VS Code — that's a fast pipe.

---

## 6. Run pyKMC

```bash
source ~/pykmc-install/pykmc/activate.sh
mpirun -n 8 python -m pykmc -in input.in
```

In your `input.in`, set the pARTn library path (use the WSL absolute path, not a Windows path):

```ini
[pARTn]
path_artnso = /home/pykmc/pykmc-install/pykmc/artn-plugin/lib/libartn-lmp.so
```

---

## 7. Editing pyKMC code from Windows

Install **Visual Studio Code** on Windows, then add the **WSL extension** (Microsoft, ID `ms-vscode-remote.remote-wsl`). To open your pyKMC source in VS Code from inside WSL:

```bash
cd ~/pykmc-install/pykmc/pyKMC
code .
```

VS Code launches on Windows with a WSL backend. Terminals, debuggers, linters, and Python interpreters all run inside Ubuntu; only the editor UI is on Windows. The first launch installs a small VS Code server inside WSL automatically.

---

## 8. Day-to-day usage

Open a pyKMC shell from Windows anytime:

```powershell
wsl -d pykmc
```

Or pin a Windows Terminal profile pointing at `wsl.exe -d pykmc`.

To shut WSL down (frees RAM):

```powershell
wsl --shutdown
```

---

## Differences from Linux / macOS

| # | Topic | Windows (WSL2) | Linux | macOS |
|---|---|---|---|---|
| 1 | Host OS | Windows 10/11 | native | native |
| 2 | Compiler / MPI | run inside Ubuntu (apt) | apt / dnf | brew |
| 3 | pyKMC build | unchanged Linux script inside WSL | Linux script | macOS script |
| 4 | Work tree location | `~/` inside WSL ext4 (NOT `/mnt/c`) | `~/` | `~/` |
| 5 | VHD relocation | required on Win10 (export/import); optional on Win11 (`--location`) | N/A | N/A |
| 6 | Library format | `.so` (Linux native) | `.so` | `.dylib` |
| 7 | Editor integration | VS Code + WSL extension | native | native |

---

## Troubleshooting

### `wsl --install` fails with "WslRegisterDistribution failed with error: 0x80370102"
Virtualization is disabled. Enable VT-x/AMD-V in BIOS, then reboot.

### After reboot, `wsl` still prints help text only
The first `dism` step succeeded but `VirtualMachinePlatform` may not have been enabled. Re-run both `dism` commands in admin PowerShell, then reboot again.

### `wsl --update` fails
On Win10 the kernel is updated by `wsl --update`; if it fails, download the manual installer from <https://aka.ms/wsl2kernel>.

### LAMMPS build runs out of memory
WSL2 caps RAM at 50% of host by default. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
processors=8
```

Then `wsl --shutdown` and re-launch.

### Slow `apt` or compile
Likely your work tree is under `/mnt/c/...`. Move it to `~/` and rebuild.
