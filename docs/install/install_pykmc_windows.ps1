#Requires -Version 5.1
<#
.SYNOPSIS
    Install pyKMC on Windows via WSL2 + Ubuntu.

.DESCRIPTION
    Sets up WSL2, installs Ubuntu 24.04, relocates the distro VHD to a fast drive,
    then runs the project's existing install_pykmc_linux.sh inside the new distro.

    The script is resumable: on first run it enables Windows features and asks
    you to reboot; on the second run it picks up at distro install. Re-running
    after a successful install is a no-op.

.PARAMETER InstallPath
    Folder on the host where the WSL distro VHD will live.
    Default: P:\WSL\Ubuntu
    Choose a fast drive (NVMe SSD) with >=30 GB free.

.PARAMETER DistroName
    Name to register the WSL distro under. Default: pykmc

.PARAMETER UbuntuRelease
    Ubuntu release to install. Default: Ubuntu-24.04

.PARAMETER LinuxUser
    Username to create inside the Ubuntu distro. Default: pykmc

.PARAMETER SkipPyKMC
    If set, stop after Ubuntu is provisioned and do not run install_pykmc_linux.sh.
    Useful if you want to run the Linux installer manually.

.PARAMETER NoWait
    If set, skip the final 'Press Enter to exit' prompt. Use for non-interactive
    runs (CI, logged sessions). The installation summary is still written to
    install-summary.txt next to the script regardless of this flag.

.EXAMPLE
    PS> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    PS> .\install_pykmc_windows.ps1

.EXAMPLE
    PS> .\install_pykmc_windows.ps1 -InstallPath 'D:\WSL\Ubuntu' -DistroName pykmc-dev

.NOTES
    Tested on: Windows 10 Home 22H2 (build 19045+), Windows 11.
    Must be run from an elevated PowerShell (Run as Administrator) on the first invocation.
#>

[CmdletBinding()]
param(
    [string]$InstallPath = 'P:\WSL\Ubuntu',
    [string]$DistroName = 'pykmc',
    [string]$UbuntuRelease = 'Ubuntu-24.04',
    [string]$LinuxUser = 'pykmc',
    [switch]$SkipPyKMC,
    [switch]$NoWait
)

$ErrorActionPreference = 'Stop'

# Encoding policy:
#   Most native exes (dism, wsl --install, wsl --export, wsl --update, etc.) emit
#   UTF-8 / ASCII / OEM-CP on Win10. Leaving [Console]::OutputEncoding at its default
#   (OEM code page) keeps that output legible — including dism progress bars which
#   use OEM box-drawing chars.
#
#   ONLY `wsl --status` and `wsl --list --verbose` emit UTF-16LE on Win10. We flip
#   the encoding to Unicode locally inside Get-WslState for those calls only, then
#   restore. See WSL_INSTALL_HANDOFF.md §4.3 for the underlying wsl.exe behavior.
$env:WSL_UTF8 = '1'  # Honored by newer WSL builds (Win11), no-op on older.

# ---------- Pretty logging (mirrors install_pykmc_linux.sh) ----------
function Step($msg) {
    Write-Host ''
    Write-Host '========================================' -ForegroundColor Yellow
    Write-Host "  $msg" -ForegroundColor Yellow
    Write-Host '========================================' -ForegroundColor Yellow
    Write-Host ''
}
function Ok($msg)   { Write-Host "[OK] $msg"   -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

# ---------- Helpers ----------
function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p  = [Security.Principal.WindowsPrincipal]::new($id)
    $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-WslState {
    # Returns a hashtable describing what is already done so we can resume.
    $state = @{
        WslFeature        = $null
        VmpFeature        = $null
        WslCommandWorks   = $false
        DistroExists      = $false
        DistroVersion     = $null
        DistroAtPath      = $false
    }
    try {
        $state.WslFeature = (Get-WindowsOptionalFeature -Online -FeatureName 'Microsoft-Windows-Subsystem-Linux' -ErrorAction Stop).State
        $state.VmpFeature = (Get-WindowsOptionalFeature -Online -FeatureName 'VirtualMachinePlatform' -ErrorAction Stop).State
    } catch {
        Warn "Could not read feature state (need admin); assuming not enabled."
    }
    # `wsl --status` returns help text on a system where WSL is dormant; treat 'WSL version' marker as 'works'.
    # NOTE: do NOT use 2>&1 here. In PS 5.1, native-command stderr is wrapped as NativeCommandError
    # ErrorRecords; combined with $ErrorActionPreference='Stop' that terminates the script. Worse,
    # wsl.exe writes some messages (e.g. "WSL is finishing an upgrade...") to stderr as ASCII while
    # stdout is UTF-16LE, so merging the streams produces garbage characters. We only need stdout.
    #
    # Encoding flip: `wsl --status` and `wsl --list --verbose` are the two wsl.exe subcommands that
    # emit UTF-16LE on Win10. Save the prior console encoding, flip to Unicode for these calls,
    # then restore — so the *rest* of the script's native-exe output (dism, wsl --install, etc.)
    # keeps using the default OEM encoding and stays legible.
    $prevEnc = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    try {
        $statusOut = (& wsl.exe --status 2>$null | Out-String)
        if ($statusOut -match 'Default Version|WSL version|Default Distribution') {
            $state.WslCommandWorks = $true
        }
        $listOut = (& wsl.exe --list --verbose 2>$null | Out-String)
        if ($listOut -match [regex]::Escape($DistroName)) {
            $state.DistroExists = $true
            if ($listOut -match "$([regex]::Escape($DistroName))\s+\S+\s+(\d)") {
                $state.DistroVersion = [int]$Matches[1]
            }
        }
    } finally {
        [Console]::OutputEncoding = $prevEnc
    }
    $state.DistroAtPath = Test-Path (Join-Path $InstallPath 'ext4.vhdx')
    return $state
}

function Invoke-Wsl {
    param([Parameter(ValueFromRemainingArguments=$true)]$Args)
    & wsl.exe @Args
    if ($LASTEXITCODE -ne 0) { Fail "wsl @Args failed with exit code $LASTEXITCODE" }
}

# =========================================================================
# Phase 0: Preflight
# =========================================================================
Step 'Phase 0 — Preflight: Windows version, virtualization, free space'

$os = Get-CimInstance Win32_OperatingSystem
$build = [int]$os.BuildNumber
Write-Host "Windows: $($os.Caption) build $build"

if ($build -lt 19041) {
    Fail "Windows build $build is too old for WSL2. Need build >= 19041 (Win10 2004 or newer / Win11)."
}
$IsWin11 = ($build -ge 22000)
if ($IsWin11) { Ok "Windows 11 detected — can use 'wsl --install --location'" }
else          { Ok "Windows 10 detected — will use export/unregister/import to relocate VHD" }

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
if (-not $cpu.VirtualizationFirmwareEnabled) {
    Fail 'CPU virtualization is disabled in BIOS/UEFI. Enable VT-x/AMD-V/SVM and re-run.'
}
Ok 'Virtualization enabled in firmware'

# Validate install path drive has enough free space.
$drive = (Split-Path -Qualifier $InstallPath).TrimEnd(':')
$vol   = Get-Volume -DriveLetter $drive -ErrorAction Stop
$freeGb = [math]::Round($vol.SizeRemaining / 1GB, 1)
Write-Host "Target drive ${drive}: $freeGb GB free"
if ($freeGb -lt 30) {
    Fail "Drive $drive has only $freeGb GB free; need >= 30 GB. Pick another drive with -InstallPath."
}
Ok "Target drive has $freeGb GB free"

# Admin check (only required for Phase 1).
$isAdmin = Test-Admin
$state = Get-WslState
Write-Host ''
Write-Host "Current state:"
Write-Host "  WSL feature           : $($state.WslFeature)"
Write-Host "  VirtualMachinePlatform: $($state.VmpFeature)"
Write-Host "  wsl command works     : $($state.WslCommandWorks)"
Write-Host "  Distro '$DistroName' exists: $($state.DistroExists) (v$($state.DistroVersion))"
Write-Host "  VHD at $InstallPath   : $($state.DistroAtPath)"

# =========================================================================
# Phase 1: Enable Windows features (admin + reboot)
# =========================================================================
# Gate solely on the optional-feature state. Do NOT include WslCommandWorks here:
# `wsl --status` can transiently return empty stdout (e.g. while a kernel update is
# still finalizing post-Phase-2), which would falsely re-trigger Phase 1 and
# prompt for a pointless second reboot. If features are Enabled, Phase 1 is done.
$needFeatures = ($state.WslFeature -ne 'Enabled') -or ($state.VmpFeature -ne 'Enabled')
if ($needFeatures) {
    Step 'Phase 1 — Enable WSL + VirtualMachinePlatform features'
    if (-not $isAdmin) {
        Fail "This phase requires admin rights. Re-launch PowerShell as Administrator and re-run this script."
    }

    & dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Host
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { Fail "dism enable WSL failed: $LASTEXITCODE" }

    & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Host
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { Fail "dism enable VirtualMachinePlatform failed: $LASTEXITCODE" }

    Ok 'Windows features enabled'
    Warn 'A REBOOT is required before WSL2 can run. After rebooting, re-run this script (admin not required after this point).'
    $ans = Read-Host 'Reboot now? [y/N]'
    if ($ans -match '^[Yy]') {
        Restart-Computer -Force
    }
    exit 0
} else {
    Ok 'WSL features already enabled'
}

# =========================================================================
# Phase 2: Set WSL2 default version + update kernel
# =========================================================================
Step 'Phase 2 — Configure WSL2 default and update kernel'

& wsl.exe --set-default-version 2 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail 'wsl --set-default-version 2 failed' }
Ok 'WSL default version = 2'

& wsl.exe --update | Out-Host
if ($LASTEXITCODE -ne 0) { Warn "wsl --update returned $LASTEXITCODE (may be benign if already current)" }
Ok 'WSL kernel updated (or already current)'

# =========================================================================
# Phase 3: Install Ubuntu and relocate VHD
# =========================================================================
$state = Get-WslState
if ($state.DistroExists -and $state.DistroAtPath) {
    Ok "Distro '$DistroName' already exists at $InstallPath — skipping install"
} else {
    Step "Phase 3 — Install $UbuntuRelease and relocate to $InstallPath"

    New-Item -ItemType Directory -Force -Path $InstallPath | Out-Null

    if ($IsWin11) {
        # Win11 path: install directly to the chosen location.
        & wsl.exe --install -d $UbuntuRelease --location $InstallPath --name $DistroName --no-launch | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "wsl --install failed: $LASTEXITCODE" }
    } else {
        # Win10 path: install, export, unregister, import.
        Write-Host "Installing $UbuntuRelease (this opens the Microsoft Store / downloads the appx)..."
        & wsl.exe --install -d $UbuntuRelease --no-launch | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "wsl --install -d $UbuntuRelease failed: $LASTEXITCODE" }

        $tar = Join-Path $InstallPath "$UbuntuRelease.tar"
        Write-Host "Exporting distro to $tar..."
        & wsl.exe --export $UbuntuRelease $tar | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "wsl --export failed: $LASTEXITCODE" }

        Write-Host "Unregistering original $UbuntuRelease entry..."
        & wsl.exe --unregister $UbuntuRelease | Out-Host

        Write-Host "Re-importing as '$DistroName' at $InstallPath..."
        & wsl.exe --import $DistroName $InstallPath $tar --version 2 | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail "wsl --import failed: $LASTEXITCODE" }

        Remove-Item $tar -Force -ErrorAction SilentlyContinue
    }
    Ok "Distro '$DistroName' installed at $InstallPath"
}

# =========================================================================
# Phase 4: Linux-side bootstrap (user, wsl.conf, pyKMC install)
# =========================================================================
# All bash work is delegated to setup_pykmc_distro.sh which lives next to this
# script. This sidesteps PS 5.1 here-string / redirection-operator parser quirks
# entirely and keeps the bash logic readable in a real .sh file.
$setupSh = Join-Path $PSScriptRoot 'setup_pykmc_distro.sh'
if (-not (Test-Path $setupSh)) {
    Fail "Cannot find setup_pykmc_distro.sh next to this script (looked at $setupSh)."
}

# Build the WSL-visible path: C:\foo\bar.sh -> /mnt/c/foo/bar.sh
$drv = $setupSh.Substring(0,1).ToLower()
$wslSetupPath = '/mnt/' + $drv + ($setupSh.Substring(2) -replace '\\','/')

$mode = if ($SkipPyKMC) { 'skip-pykmc' } else { 'full' }

Step "Phase 4 — Provision user '$LinuxUser' and run pyKMC installer (mode: $mode)"

& wsl.exe -d $DistroName -u root -- bash $wslSetupPath $LinuxUser $mode
if ($LASTEXITCODE -ne 0) {
    Fail "setup_pykmc_distro.sh failed inside WSL (exit $LASTEXITCODE). For diagnostics:`n   wsl -d $DistroName -- bash -lc 'tail -100 ~/pykmc-install/install.log'"
}
Ok 'Linux-side bootstrap complete'

& wsl.exe --terminate $DistroName | Out-Null

# =========================================================================
# Done
# =========================================================================
# Build a structured summary, write it to a sibling file (so it survives even if
# the user's terminal scroll buffer is shorter than the install output), then
# print it green and block on Enter so it cannot scroll off-screen unnoticed.
$summary = @"
========================================
  Installation complete!
========================================

WSL distro      : $DistroName  (version 2)
VHD location    : $InstallPath\ext4.vhdx
Linux user      : $LinuxUser
pyKMC location  : /home/$LinuxUser/pykmc-install/pykmc/

To use pyKMC:
  wsl -d $DistroName
  source ~/pykmc-install/pykmc/activate.sh
  mpirun -n 8 python -m pykmc -in input.in

For inputs that load the pARTn plugin, set in input.in:
  [pARTn]
  path_artnso = /home/$LinuxUser/pykmc-install/pykmc/artn-plugin/lib/libartn-lmp.so

  This MUST be an absolute path. LAMMPS's `plugin load` does not expand `~` or
  environment variables; a `~/...` path will silently fail to load and event
  searches will die with "Unrecognized fix style 'artn'".

Edit code from Windows with VS Code + WSL extension:
  wsl -d $DistroName
  cd ~/pykmc-install/pykmc/pyKMC
  code .
"@

$summaryPath = Join-Path $PSScriptRoot 'install-summary.txt'
$summary | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host ''
Write-Host $summary -ForegroundColor Green
Write-Host ''
Write-Host "(Summary saved to $summaryPath -- run 'type install-summary.txt' any time)" -ForegroundColor Cyan
Write-Host ''

if (-not $NoWait) {
    [void](Read-Host 'Press Enter to exit')
}
