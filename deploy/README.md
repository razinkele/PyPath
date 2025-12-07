# PyPath Deployment to laguna.ku.lt

This directory contains scripts for deploying the PyPath Shiny for Python application to the existing Shiny Server on laguna.ku.lt.

## Server Configuration

- **Server**: laguna.ku.lt
- **Shiny Server**: Already running
- **App Location**: `/srv/shiny-server/pypath`
- **URL**: `http://laguna.ku.lt/pypath`

## Deployment Steps

### Step 1: Create Deployment Package (Windows)

Open PowerShell and run:

```powershell
cd c:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\PyPath\deploy
.\prepare_package.ps1
```

This creates `pypath_deploy.tar.gz` in the deploy directory.

### Step 2: Upload to Server

```bash
scp pypath_deploy.tar.gz user@laguna.ku.lt:/tmp/
```

Replace `user` with your username on laguna.ku.lt.

### Step 3: Deploy on Server

SSH into the server:

```bash
ssh user@laguna.ku.lt
```

Extract and run the deployment:

```bash
cd /tmp
tar -xzf pypath_deploy.tar.gz
cd pypath_deploy
chmod +x deploy.sh
sudo ./deploy.sh
```

### Step 4: Configure Shiny Server (First Time Only)

If this is the first deployment, add the following to `/etc/shiny-server/shiny-server.conf` inside the `server` block:

```text
  location /pypath {
    site_dir /srv/shiny-server/pypath;
    python /srv/shiny-server/pypath/venv/bin/python;
    log_dir /var/log/shiny-server/pypath;
    directory_index on;
  }
```

Then restart Shiny Server:

```bash
sudo systemctl restart shiny-server
```

### Step 5: Verify Deployment

Open in browser: `http://laguna.ku.lt/pypath`

## Updating an Existing Deployment

```bash
# Upload new package
scp pypath_deploy.tar.gz user@laguna.ku.lt:/tmp/

# On server
ssh user@laguna.ku.lt
cd /tmp
tar -xzf pypath_deploy.tar.gz
cd pypath_deploy
sudo ./deploy.sh --update
```

## Files in This Directory

| File | Description |
|------|-------------|
| `prepare_package.ps1` | Windows script to create deployment tarball |
| `deploy.sh` | Main deployment script for Shiny Server |
| `requirements.txt` | Python dependencies |
| `pypath_manage.sh` | Service management helper |

## Directory Structure on Server

```text
/srv/shiny-server/
└── pypath/
    ├── app.py              # Main Shiny app entry point
    ├── app/                # Application modules
    │   ├── pages/          # Page modules
    │   └── static/         # Static assets
    ├── src/                # PyPath library
    │   └── pypath/
    ├── venv/               # Python virtual environment
    └── data/               # User data directory
```

## Useful Commands

```bash
# Check Shiny Server status
sudo systemctl status shiny-server

# View Shiny Server logs
sudo tail -f /var/log/shiny-server.log

# View PyPath app logs
sudo tail -f /var/log/shiny-server/pypath/*.log

# Restart Shiny Server
sudo systemctl restart shiny-server

# Use management script (after copying to server)
./pypath_manage.sh status
./pypath_manage.sh logs
./pypath_manage.sh restart
```

## Troubleshooting

1. **App not loading**: Check Shiny Server logs and ensure Python path is correct
2. **Permission denied**: Ensure shiny user owns the app directory
3. **Module not found**: Verify virtual environment has all dependencies
4. **502 Bad Gateway**: Restart Shiny Server with `sudo systemctl restart shiny-server`
