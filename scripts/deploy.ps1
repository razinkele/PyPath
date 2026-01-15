<#
PowerShell deploy helper for manual deployment to a remote Shiny server.
Requires OpenSSH client installed (ssh & scp) on Windows.
Usage examples:
  .\scripts\deploy.ps1 -Host laguna.ku.lt -User razinka -Path /home/razinka/shiny/pypath -Key C:\Users\you\.ssh\pypath_deploy
  .\scripts\deploy.ps1 -Host laguna.ku.lt -User razinka -Path /srv/shiny-server/pypath -Key C:\id_rsa -Restart
#>

param(
    [Parameter(Mandatory = $true)] [string] $Host,
    [Parameter(Mandatory = $true)] [string] $User,
    [Parameter(Mandatory = $true)] [string] $Path,
    [string] $Key = "$env:USERPROFILE\.ssh\id_rsa",
    [switch] $Restart,
    [switch] $DryRun
)

function Show-Help {
    @"
Usage: deploy.ps1 -Host <host> -User <user> -Path <remote path> [-Key <ssh key>] [-Restart] [-DryRun]
Example:
  .\scripts\deploy.ps1 -Host laguna.ku.lt -User razinka -Path /home/razinka/shiny/pypath -Key C:\Users\you\.ssh\pypath_deploy
"@
}

if ($PSBoundParameters.ContainsKey('Host') -eq $false -or $PSBoundParameters.ContainsKey('User') -eq $false -or $PSBoundParameters.ContainsKey('Path') -eq $false) {
    Show-Help
    exit 2
}

# Ensure ssh/scp are available
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error "ssh command not found. Install OpenSSH client or use WSL."
    exit 3
}
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Error "scp command not found. Install OpenSSH client or use WSL."
    exit 3
}

# Create temporary zip of repository root (omits .git, tests, .github)
$Tmp = [System.IO.Path]::GetTempFileName() + ".zip"
$Exclude = @('.git', 'tests', '.github', 'venv', 'env', '__pycache__')
$Items = Get-ChildItem -Path . -Force | Where-Object { $Exclude -notcontains $_.Name }

Write-Host "Creating zip archive $Tmp ..."
Compress-Archive -Path $Items -DestinationPath $Tmp -Force

$RemoteTmp = "/tmp/$(Split-Path -Leaf $Tmp)"

$SCP_CMD = "scp -v -i `"$Key`" `"$Tmp`" $User@$Host:$RemoteTmp"
Write-Host "Uploading archive to $Host:$RemoteTmp ..."
if ($DryRun) { Write-Host "DRY RUN: $SCP_CMD"; exit 0 }

$scpexit = & scp -i $Key $Tmp "$User@$Host:$RemoteTmp"
if ($LASTEXITCODE -ne 0) { Write-Error "scp failed"; Remove-Item $Tmp -ErrorAction SilentlyContinue; exit 4 }

# Extract and move into place on remote
$SSH_CMD = @(
    "mkdir -p '$Path'",
    "unzip -o '$RemoteTmp' -d '$Path'",
    "rm -f '$RemoteTmp'"
) -join "; "

Write-Host "Extracting on remote host..."
ssh -i $Key $User@$Host $SSH_CMD
if ($LASTEXITCODE -ne 0) { Write-Error "Remote extraction failed"; Remove-Item $Tmp -ErrorAction SilentlyContinue; exit 5 }

if ($Restart) {
    Write-Host "Attempting to restart Shiny Server on remote host (may require sudo) ..."
    ssh -i $Key $User@$Host "sudo systemctl restart shiny-server || sudo service shiny-server restart || true"
}

Remove-Item $Tmp -ErrorAction SilentlyContinue
Write-Host "Deploy finished."
exit 0
