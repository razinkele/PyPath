#!/bin/bash
#
# PyPath Shiny App Deployment Script for laguna.ku.lt
#
# This script deploys the PyPath Shiny for Python application to an
# existing Shiny Server installation on laguna.ku.lt.
#
# The repository is a monorepo with two packages:
#   packages/pypath/       -> pypath-ewe  (core algorithms)
#   packages/pypath-shiny/ -> pypath-shiny (Shiny web frontend)
#
# Usage:
#   sudo ./deploy.sh           # Fresh install
#   sudo ./deploy.sh --update  # Update existing installation
#
# Requirements:
#   - Python 3.10+
#   - Existing Shiny Server installation
#   - sudo privileges
#

set -e  # Exit on error

# =============================================================================
# Configuration - Modify these settings as needed
# =============================================================================

APP_NAME="pypath"
# Shiny Server app directory
SHINY_SERVER_DIR="/srv/shiny-server"
APP_DIR="${SHINY_SERVER_DIR}/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
APP_USER="shiny"
APP_GROUP="shiny"
PYTHON_VERSION="python3"

# Server hostname for configuration
SERVER_HOSTNAME="laguna.ku.lt"

# Log file location (Shiny Server manages logs)
LOG_DIR="/var/log/shiny-server/${APP_NAME}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_python() {
    if ! command -v ${PYTHON_VERSION} &> /dev/null; then
        log_error "Python ${PYTHON_VERSION} not found. Please install it first."
        log_info "Try: sudo apt install python3 python3-venv python3-dev"
        exit 1
    fi
    log_info "Found $(${PYTHON_VERSION} --version)"
}

check_shiny_server() {
    if ! systemctl is-active --quiet shiny-server; then
        log_warn "Shiny Server is not running. Will attempt to start after deployment."
    else
        log_info "Shiny Server is running"
    fi

    if [[ ! -d "${SHINY_SERVER_DIR}" ]]; then
        log_error "Shiny Server directory not found: ${SHINY_SERVER_DIR}"
        exit 1
    fi
}

# =============================================================================
# Main Deployment Functions
# =============================================================================

create_user() {
    if ! id "${APP_USER}" &>/dev/null; then
        log_info "Creating service user: ${APP_USER}"
        useradd --system --no-create-home --shell /bin/false ${APP_USER}
    else
        log_info "User ${APP_USER} already exists"
    fi
}

create_directories() {
    log_info "Creating application directories..."

    mkdir -p "${APP_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${APP_DIR}/data"

    # Set ownership to shiny user (Shiny Server requirement)
    chown -R ${APP_USER}:${APP_GROUP} "${APP_DIR}"
    chown -R ${APP_USER}:${APP_GROUP} "${LOG_DIR}" 2>/dev/null || true
}

setup_virtualenv() {
    log_info "Setting up Python virtual environment..."

    if [[ -d "${VENV_DIR}" ]]; then
        if [[ "$1" == "--reuse" ]]; then
            log_info "Reusing existing virtual environment"
            return
        fi
        log_warn "Virtual environment exists, removing old one..."
        rm -rf "${VENV_DIR}"
    fi

    ${PYTHON_VERSION} -m venv "${VENV_DIR}"

    # Activate and upgrade pip
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip wheel setuptools
    deactivate
}

copy_application() {
    log_info "Copying application files..."

    # Copy monorepo packages directory
    if [[ -d "packages" ]]; then
        cp -r packages "${APP_DIR}/"
    else
        log_error "packages/ directory not found. Are you in the PyPath repo root?"
        exit 1
    fi

    # Remove test files and caches from deployed copy to save space
    find "${APP_DIR}/packages" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
    find "${APP_DIR}/packages" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${APP_DIR}/packages" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find "${APP_DIR}/packages" -name "*.pyc" -delete 2>/dev/null || true

    # Generate app.py entry point with sys.path fix for Shiny Server
    # (Shiny Server uses su --login which prevents .pth editable installs from working)
    cat > "${APP_DIR}/app.py" << 'APPENTRY'
import sys
from pathlib import Path

app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir / "packages" / "pypath-shiny" / "src"))
sys.path.insert(0, str(app_dir / "packages" / "pypath" / "src"))

from pypath_shiny.app import app

__all__ = ["app"]
APPENTRY

    # Set ownership and permissions
    chown -R ${APP_USER}:${APP_GROUP} "${APP_DIR}"
    chmod -R 755 "${APP_DIR}"
    find "${APP_DIR}" -type f -name "*.py" -exec chmod 644 {} \;
}

install_packages() {
    log_info "Installing pypath packages into virtual environment..."

    source "${VENV_DIR}/bin/activate"

    # Install core package first (pypath-shiny depends on pypath-ewe)
    pip install -e "${APP_DIR}/packages/pypath"
    # Install frontend package
    pip install -e "${APP_DIR}/packages/pypath-shiny"

    deactivate

    log_info "Packages installed successfully"
}

configure_shiny_server() {
    log_info "Configuring Shiny Server..."

    SHINY_CONF="/etc/shiny-server/shiny-server.conf"

    # Check if pypath location already exists in config
    if grep -q "location /pypath" "${SHINY_CONF}" 2>/dev/null; then
        log_info "PyPath location already configured in Shiny Server"
    else
        log_info "Adding PyPath configuration to Shiny Server..."

        # Create a config snippet
        cat > "${APP_DIR}/shiny-server-pypath.conf" << EOF
# PyPath Shiny Application
# Add this block inside the 'server' section of /etc/shiny-server/shiny-server.conf

  location /pypath {
    site_dir ${APP_DIR};
    python ${VENV_DIR}/bin/python;
    log_dir ${LOG_DIR};
    directory_index on;
  }
EOF

        log_warn "Please add the following to ${SHINY_CONF}:"
        cat "${APP_DIR}/shiny-server-pypath.conf"
        log_info "Config snippet saved to: ${APP_DIR}/shiny-server-pypath.conf"
    fi
}

restart_shiny_server() {
    log_info "Restarting Shiny Server..."

    if systemctl is-active --quiet shiny-server; then
        systemctl restart shiny-server
        sleep 3

        if systemctl is-active --quiet shiny-server; then
            log_info "Shiny Server restarted successfully"
        else
            log_error "Shiny Server failed to restart"
            log_error "Check logs: sudo journalctl -u shiny-server -n 50"
            exit 1
        fi
    else
        log_warn "Shiny Server not running, attempting to start..."
        systemctl start shiny-server
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."

    # Check if app.py exists
    if [[ ! -f "${APP_DIR}/app.py" ]]; then
        log_error "app.py not found in ${APP_DIR}"
        exit 1
    fi

    # Check if packages directory exists
    if [[ ! -d "${APP_DIR}/packages/pypath" ]] || [[ ! -d "${APP_DIR}/packages/pypath-shiny" ]]; then
        log_error "packages/ directory incomplete in ${APP_DIR}"
        exit 1
    fi

    # Check if virtual environment works with both packages
    if ! "${VENV_DIR}/bin/python" -c "import pypath; import pypath_shiny" 2>/dev/null; then
        log_warn "Could not verify Python imports. Check virtual environment."
    else
        log_info "Python imports verified (pypath + pypath_shiny)"
    fi

    # Check permissions
    if [[ $(stat -c '%U' "${APP_DIR}") != "${APP_USER}" ]]; then
        log_warn "App directory not owned by ${APP_USER}"
        chown -R ${APP_USER}:${APP_GROUP} "${APP_DIR}"
    fi

    log_info "Deployment verification complete"
}

show_completion_info() {
    log_info "Deployment complete!"

    echo
    echo "============================================="
    echo "  PyPath deployed to Shiny Server"
    echo "============================================="
    echo
    echo "  Application URL: http://${SERVER_HOSTNAME}/pypath"
    echo "  App directory:   ${APP_DIR}"
    echo "  Log directory:   ${LOG_DIR}"
    echo
    echo "  Useful commands:"
    echo "    sudo systemctl status shiny-server"
    echo "    sudo tail -f ${LOG_DIR}/*.log"
    echo "    sudo tail -f /var/log/shiny-server.log"
    echo
}

update_application() {
    log_info "Updating existing installation..."

    # Backup current packages
    if [[ -d "${APP_DIR}/packages" ]]; then
        mv "${APP_DIR}/packages" "${APP_DIR}/packages.bak.$(date +%Y%m%d%H%M%S)"
    fi

    # Copy new files
    copy_application

    # Reuse existing venv, reinstall packages
    setup_virtualenv --reuse
    install_packages

    # Fix ownership
    chown -R ${APP_USER}:${APP_GROUP} "${APP_DIR}"

    # Restart Shiny Server
    restart_shiny_server

    log_info "Update complete!"
    show_completion_info
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    echo "=============================================="
    echo "  PyPath Shiny App Deployment"
    echo "  Target: ${SERVER_HOSTNAME} (Shiny Server)"
    echo "=============================================="
    echo

    check_root
    check_python
    check_shiny_server

    if [[ "$1" == "--update" ]]; then
        update_application
    else
        create_user
        create_directories
        setup_virtualenv
        copy_application
        install_packages
        configure_shiny_server
        verify_deployment
        restart_shiny_server
        show_completion_info
    fi
}

main "$@"
