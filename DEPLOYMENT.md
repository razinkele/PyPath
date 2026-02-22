# Deployment guide — manual deploy scripts and GitHub Secrets

This document describes how to perform manual deployments to the existing Shiny server on `laguna.ku.lt` and how to prepare the server and GitHub repository for automated deployments.

## Architecture

PyPath is a **monorepo** with two packages:

| Package | Path | PyPI Name | Import |
|---------|------|-----------|--------|
| Core algorithms | `packages/pypath/` | `pypath-ewe` | `import pypath` |
| Shiny frontend | `packages/pypath-shiny/` | `pypath-shiny` | `import pypath_shiny` |

Both are installed from source into a venv on the server using `pip install -e`. Dependencies are managed by `pyproject.toml` in each package (not `requirements.txt`).

## 1) Recommended workflow (manual)

1. Create a dedicated deploy SSH key on your local machine:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/pypath_deploy -C "deploy@pypath"
   ```

2. Copy the public key to the remote server for the `razinka` user:

   ```bash
   ssh-copy-id -i ~/.ssh/pypath_deploy.pub razinka@laguna.ku.lt
   ```

3. Verify you can log in (and optionally that sudo works if you plan to restart Shiny Server):

   ```bash
   ssh -i ~/.ssh/pypath_deploy razinka@laguna.ku.lt
   # check sudo (if needed): sudo -n true || echo "No passwordless sudo"
   ```

4. Use the helper scripts in the `scripts/` directory to deploy from your machine:

   - Linux/macOS (rsync):
     ```bash
     ./scripts/deploy.sh --host laguna.ku.lt --user razinka --path /srv/shiny-server/pypath --key ~/.ssh/pypath_deploy
     # add --restart to attempt restarting Shiny Server (requires sudo on remote)
     ```

   - Windows (PowerShell):
     ```powershell
     .\scripts\deploy.ps1 -Host laguna.ku.lt -User razinka -Path /srv/shiny-server/pypath -Key C:\Users\you\.ssh\pypath_deploy
     # add -Restart to attempt restarting Shiny Server
     ```

   After rsync/upload, SSH into the server and install packages:

   ```bash
   ssh razinka@laguna.ku.lt
   TARGET=/srv/shiny-server/pypath

   # Create venv if first time
   python3 -m venv $TARGET/venv

   # Install both packages (order matters: core first)
   source $TARGET/venv/bin/activate
   pip install -e $TARGET/packages/pypath
   pip install -e $TARGET/packages/pypath-shiny
   deactivate

   # Fix ownership
   sudo chown -R shiny:shiny $TARGET
   sudo systemctl restart shiny-server
   ```

Notes:
- The scripts exclude `.git`, `.github`, `.claude`, `tests`, caches, and build artifacts by default.
- Dependencies are resolved from `pyproject.toml` during `pip install -e`.

## 2) GitHub Secrets (for automated deploys)

Add the following repository secrets (Repository Settings > Secrets > Actions):

- `SSH_PRIVATE_KEY` — private key content for the deploy key
- `DEPLOY_HOST` — e.g., `laguna.ku.lt`
- `DEPLOY_USER` — e.g., `razinka`
- `DEPLOY_PATH` — remote path, e.g., `/srv/shiny-server/pypath`
- `RESTART_AFTER_DEPLOY` — (optional) set to `true` to auto-restart Shiny Server

The workflow (`.github/workflows/deploy.yml`) will:
1. Rsync the repo (excluding tests, caches, build artifacts)
2. Create a venv if missing
3. Generate `app.py` with `sys.path` entries (see note below)
4. `pip install -e` both packages on the remote server
5. Fix file ownership to `shiny:shiny`
6. Optionally restart Shiny Server

## 3) Server-side notes

- Target directory: `/srv/shiny-server/pypath/`
- The `shiny` user must own the app directory
- Shiny Server looks for `app.py` at the root
- **Important:** Shiny Server uses `su --login` to switch to the `shiny` user, which resets the environment and prevents `.pth` files (created by `pip install -e`) from being processed. The generated `app.py` includes explicit `sys.path.insert()` calls pointing to `packages/pypath/src/` and `packages/pypath-shiny/src/` to work around this.
- Typical Shiny Server restart commands (may require sudo):
  - `sudo systemctl restart shiny-server`
  - `sudo service shiny-server restart`

## 4) Server directory layout

```text
/srv/shiny-server/pypath/
├── app.py                          # sys.path fix + from pypath_shiny.app import app
├── packages/
│   ├── pypath/                     # pypath-ewe source
│   │   ├── src/pypath/
│   │   └── pyproject.toml
│   └── pypath-shiny/               # pypath-shiny source
│       ├── src/pypath_shiny/
│       └── pyproject.toml
├── venv/                           # Both packages pip-installed here
│   └── bin/python
├── data/                           # Optional runtime data
└── shiny-server-pypath.conf        # Config snippet for admin
```

## 5) Alternative: package-based deploy

Instead of the scripts in `scripts/`, you can use the tarball-based workflow in `deploy/`:

```powershell
# On Windows: create deployment tarball
.\deploy\prepare_package.ps1

# Upload and deploy on server
scp pypath_deploy.tar.gz razinka@laguna.ku.lt:/tmp/
ssh razinka@laguna.ku.lt
cd /tmp && tar -xzf pypath_deploy.tar.gz && cd pypath_deploy
sudo ./deploy.sh        # fresh install
sudo ./deploy.sh --update  # update existing
```

See `deploy/README.md` for full details.
