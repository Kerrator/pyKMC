#!/bin/bash
#
# Inside-WSL bootstrap, called by install_pykmc_windows.ps1 after the Ubuntu
# distro is provisioned and relocated. Idempotent.
#
# Steps:
#   1. Create the default Linux user (passwordless sudo) if missing.
#   2. Write /etc/wsl.conf so the distro launches into that user with systemd.
#   3. As the user, fetch the project's install_pykmc_linux.sh and run it.
#
# Arguments:
#   $1 = Linux username to create (e.g. "pykmc")
#   $2 = (optional) "skip-pykmc" to stop after wsl.conf is written
#
# This script must run as root inside WSL. The PowerShell wrapper invokes it via:
#   wsl -d <distro> -u root -- bash /mnt/<host-path-to-script> <user> [skip-pykmc]
#
set -euo pipefail

LINUX_USER="${1:?usage: setup_pykmc_distro.sh <linux_user> [skip-pykmc]}"
MODE="${2:-full}"

# ---------- 1. Create user ----------
if id -u "$LINUX_USER" >/dev/null 2>&1; then
    echo "[OK] User '$LINUX_USER' already exists"
else
    useradd -m -s /bin/bash -G sudo "$LINUX_USER"
    passwd -d "$LINUX_USER"
    echo "$LINUX_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$LINUX_USER"
    chmod 0440 "/etc/sudoers.d/$LINUX_USER"
    echo "[OK] Created user '$LINUX_USER' with passwordless sudo"
fi

# ---------- 2. /etc/wsl.conf ----------
cat >/etc/wsl.conf <<EOF
[user]
default=$LINUX_USER
[boot]
systemd=true
EOF
echo "[OK] Wrote /etc/wsl.conf"

if [ "$MODE" = "skip-pykmc" ]; then
    echo "[OK] Distro provisioned. Skipping pyKMC install (caller asked for skip-pykmc)."
    exit 0
fi

# ---------- 3. Run install_pykmc_linux.sh as the user ----------
# Use a fresh download from the develop branch so users always get the latest
# verified Linux installer. To use a local copy, see install_pykmc_windows.ps1.
LINUX_INSTALLER_URL="https://raw.githubusercontent.com/hugomoison/pyKMC/develop/docs/install/install_pykmc_linux.sh"

apt-get update -y
apt-get install -y curl ca-certificates

sudo -u "$LINUX_USER" -H bash -c '
set -euo pipefail
mkdir -p ~/pykmc-install
cd ~/pykmc-install
curl -fsSLO "'"$LINUX_INSTALLER_URL"'"
chmod +x install_pykmc_linux.sh
./install_pykmc_linux.sh 2>&1 | tee install.log
'

echo "[OK] pyKMC installed inside WSL"
