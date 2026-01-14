# Deployment guide — manual deploy scripts and GitHub Secrets

This document describes how to perform manual deployments to the existing Shiny server on `laguna.ku.lt` and how to prepare the server and GitHub repository for automated deployments.

## 1) Recommended workflow (manual)

1. Create a dedicated deploy SSH key on your local machine:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/pypath_deploy -C "deploy@pypath"
   ```

2. Copy the public key to the remote server for the `razinka` user:

   ```bash
   # preferred (if ssh-copy-id available):
   ssh-copy-id -i ~/.ssh/pypath_deploy.pub razinka@laguna.ku.lt

   # or manually (on server):
   # append the public key to ~/.ssh/authorized_keys and set correct perms
   cat pypath_deploy.pub | ssh razinka@laguna.ku.lt 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'
   ```

3. Verify you can log in (and optionally that sudo works if you plan to restart Shiny Server):

   ```bash
   ssh -i ~/.ssh/pypath_deploy razinka@laguna.ku.lt
   # check sudo (if needed): sudo -n true || echo "No passwordless sudo"
   ```

4. Use the helper scripts in the `scripts/` directory to deploy from your machine:

   - Linux/macOS (rsync):
     ```bash
     ./scripts/deploy.sh --host laguna.ku.lt --user razinka --path /home/razinka/shiny/pypath --key ~/.ssh/pypath_deploy
     # add --restart to attempt restarting Shiny Server (requires sudo on remote)
     ```

   - Windows (PowerShell):
     ```powershell
     .\scripts\deploy.ps1 -Host laguna.ku.lt -User razinka -Path /home/razinka/shiny/pypath -Key C:\Users\you\.ssh\pypath_deploy
     # add -Restart to attempt restarting Shiny Server
     ```

Notes:
- The scripts exclude `.git`, `.github`, `tests`, and common virtual environment directories by default.
- The `/home/razinka/shiny/pypath` path is an example; replace with the correct target deployment folder.
- If you prefer a CI-driven approach, see section 4 for a sample GitHub Actions snippet.

## 2) GitHub Secrets (for future automated deploys)

When you want to add an automated deployment workflow (e.g., on push to `main` or on tags), add the following repository secrets (Repository Settings → Secrets → Actions):

- `SSH_PRIVATE_KEY` — private key content for the deploy key (use the private key you generated above)
- `DEPLOY_HOST` — e.g., `laguna.ku.lt`
- `DEPLOY_USER` — e.g., `razinka`
- `DEPLOY_PATH` — remote path where the app should be deployed
- `SSH_KNOWN_HOSTS` — (optional) known_hosts entry for `laguna.ku.lt` (helps avoid StrictHostKeyChecking prompts)

Keep keys scoped to a deploy-only user and avoid re-using personal keys. If the deploy action needs to restart system services and you want to avoid storing passwords, configure `razinka` with passwordless sudo for specific commands only.

## 3) Server-side notes

- Ensure the `razinka` user has write access to the target `DEPLOY_PATH` (mkdir -p as needed).
- Ensure the `~/.ssh/authorized_keys` is present and permissions are correct (700 for `~/.ssh`, 600 for the file).
- Typical Shiny Server restart commands (may require sudo):
  - `sudo systemctl restart shiny-server`
  - `sudo service shiny-server restart`

## 4) Example GitHub Actions snippet (optional)

When you are ready to automate deployments, here is a minimal example (to include in `.github/workflows/deploy.yml`):

```yaml
name: Deploy
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up SSH
        uses: webfactory/ssh-agent@v0.8.1
        with:
          ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}
      - name: Rsync deploy
        run: |
          rsync -avz --delete --exclude .git --exclude .github --exclude tests ./ ${{ secrets.DEPLOY_USER }}@${{ secrets.DEPLOY_HOST }}:${{ secrets.DEPLOY_PATH }}

# Note: to restart Shiny server, add an ssh command step which will do 'sudo systemctl restart shiny-server' if sudo is allowed for the deploy user.
```

---

If you want, I can add the above workflow now and (optionally) gate the deploy to be a manual workflow_dispatch or to run only on tags.
