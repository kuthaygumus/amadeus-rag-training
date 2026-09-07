# Windows bootstrap for the RAG training day.
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
#
# verify_setup.py is the real check and it runs everywhere. This script exists for the case that
# breaks first on a locked-down Windows laptop: Python not being there at all, so the real check
# cannot start. It needs no administrator rights and installs nothing.

$ErrorActionPreference = "Continue"
$problems = @()

function Report($label, $ok, $detail, $fix) {
    $mark = if ($ok) { "  ok  " } else { " FAIL " }
    $colour = if ($ok) { "Green" } else { "Red" }
    Write-Host $mark -ForegroundColor $colour -NoNewline
    Write-Host " $label" -NoNewline
    if ($detail) { Write-Host "  $detail" -ForegroundColor DarkGray } else { Write-Host "" }
    if (-not $ok -and $fix) { $script:problems += $fix }
}

Write-Host ""
Write-Host "RAG Training Day - Windows setup check" -ForegroundColor DarkGray
Write-Host "$([System.Environment]::OSVersion.VersionString)" -ForegroundColor DarkGray
Write-Host ""

# Python -----------------------------------------------------------------------------
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    $found = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($found) {
        $version = & $candidate --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 10) { $python = $candidate; break }
        }
    }
}
Report "Python 3.10 or newer" ($null -ne $python) $(if ($python) { (& $python --version 2>&1) }) `
    "Install Python from python.org. Tick 'Add python.exe to PATH' on the first screen. No admin rights are needed if you choose 'Install for me only'."

# Ollama -----------------------------------------------------------------------------
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
Report "Ollama installed" ($null -ne $ollama) $(if ($ollama) { $ollama.Source }) `
    "Download OllamaSetup.exe from ollama.com. It installs per-user under %LOCALAPPDATA% and does not ask for administrator rights."

$serverUp = $false
if ($ollama) {
    try {
        $version = (Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 10).version
        $serverUp = $true
        Report "Ollama is running" $true "version $version" ""
    } catch {
        Report "Ollama is running" $false "nothing answered on localhost:11434" `
            "Start it: run 'ollama serve' in a terminal, or launch Ollama from the Start menu."
    }
}

# Models -----------------------------------------------------------------------------
if ($serverUp) {
    $tags = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 30).models
    foreach ($wanted in @("qwen2.5:3b", "bge-m3")) {
        $model = $tags | Where-Object { $_.name -eq $wanted -or $_.name -eq "$wanted`:latest" } | Select-Object -First 1
        $size = if ($model) { "{0:N1} GB" -f ($model.size / 1e9) } else { "" }
        Report "model $wanted" ($null -ne $model) $size "Run: ollama pull $wanted"
    }
}

# Reachability, so a blocked network is diagnosed here rather than on the day ---------
if (-not $serverUp -or $problems.Count -gt 0) {
    try {
        $code = (Invoke-WebRequest -Uri "https://registry.ollama.ai/v2/library/bge-m3/manifests/latest" `
                 -TimeoutSec 20 -UseBasicParsing).StatusCode
        Report "can reach registry.ollama.ai" ($code -eq 200) "HTTP $code" ""
    } catch {
        Report "can reach registry.ollama.ai" $false $_.Exception.Message `
            "The model registry is not reachable from this machine. Tell the trainer before the day - you will need the models copied from a USB stick."
    }
}

Write-Host ""
if ($problems.Count -gt 0) {
    Write-Host "NOT READY" -ForegroundColor Red
    Write-Host ""
    foreach ($p in $problems) { Write-Host "  -> $p" }
    Write-Host ""
    Write-Host "Pulling the models takes a few minutes and about 3 GB." -ForegroundColor DarkGray
    Write-Host "Do it at home, not on the office network with twenty other people." -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

Write-Host "Basic checks passed. Now run the full one:" -ForegroundColor Green
Write-Host "  $python scripts\verify_setup.py"
Write-Host ""
