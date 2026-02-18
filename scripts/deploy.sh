#!/usr/bin/env bash
# Simple rsync-based deploy helper for manual deploys to a remote Shiny server.
# Usage examples:
#   ./scripts/deploy.sh --host laguna.ku.lt --user razinka --path /home/razinka/shiny/pypath --key ~/.ssh/pypath_deploy
#   ./scripts/deploy.sh --host laguna.ku.lt --user razinka --path /srv/shiny-server/pypath --key ~/.ssh/pypath_deploy --restart

set -euo pipefail

PROGNAME="$(basename "$0")"
HOST=""
USER=""
TARGET_PATH=""
KEY=""
RESTART=false
DRYRUN=false
EXCLUDES=(".git" "tests" ".github" "venv" "env" "__pycache__")

show_help() {
    cat <<-EOF
    Usage: $PROGNAME [options]

    Options:
      -h, --host HOST        Target host (e.g., laguna.ku.lt)
      -u, --user USER        Remote user (e.g., razinka)
      -p, --path PATH        Remote target directory (e.g., /home/razinka/shiny/pypath)
      -k, --key KEYFILE      SSH private key file to use
      --restart              Restart Shiny Server after deploy (uses sudo on remote)
      --dry-run              Show rsync actions without making changes
      --exclude PATTERN      Add extra rsync exclude (may be repeated)
      --help                 Show this help message

    Example:
      $PROGNAME --host laguna.ku.lt --user razinka --path /home/razinka/shiny/pypath --key ~/.ssh/pypath_deploy

EOF
}

# Simple arg parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--host)
            HOST="$2"; shift 2;;
        -u|--user)
            USER="$2"; shift 2;;
        -p|--path)
            TARGET_PATH="$2"; shift 2;;
        -k|--key)
            KEY="$2"; shift 2;;
        --restart)
            RESTART=true; shift 1;;
        --dry-run)
            DRYRUN=true; shift 1;;
        --exclude)
            EXCLUDES+=("$2"); shift 2;;
        --help)
            show_help; exit 0;;
        *)
            echo "Unknown option: $1" >&2; show_help; exit 2;;
    esac
done

if [[ -z "$HOST" || -z "$USER" || -z "$TARGET_PATH" ]]; then
    echo "Error: --host, --user and --path are required" >&2
    show_help
    exit 2
fi

SSH_OPTS=("-o" "StrictHostKeyChecking=no" "-o" "UserKnownHostsFile=/dev/null")
if [[ -n "$KEY" ]]; then
    SSH_OPTS=("-i" "$KEY" "${SSH_OPTS[@]}")
fi

RSYNC_EXCLUDES=()
for e in "${EXCLUDES[@]}"; do
    RSYNC_EXCLUDES+=("--exclude" "$e")
done

RSYNC_CMD=(rsync -avz --delete "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" ./ "$USER@$HOST:$TARGET_PATH")

if $DRYRUN; then
    echo "DRY RUN: ${RSYNC_CMD[*]}"
    ${RSYNC_CMD[@]} --dry-run
    exit 0
fi

echo "Deploying to $USER@$HOST:$TARGET_PATH ..."
# Ensure remote directory exists
ssh "${SSH_OPTS[@]}" "$USER@$HOST" "mkdir -p '$TARGET_PATH'" \
    || { echo "Failed to create remote directory" >&2; exit 3; }

# Run rsync
echo "Running rsync..."
${RSYNC_CMD[@]}

if $RESTART; then
    echo "Attempting to restart Shiny Server on remote host (may require sudo)..."
    ssh "${SSH_OPTS[@]}" "$USER@$HOST" "sudo systemctl restart shiny-server || sudo service shiny-server restart || true"
    echo "Restart command executed (may require sudo privileges)."
fi

echo "Deploy finished."
exit 0
