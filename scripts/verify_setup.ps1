# Windows bootstrap for the RAG training day.
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
#
# If that answers "running scripts is disabled on this system", the execution policy on this
# machine is set by Group Policy. The -ExecutionPolicy switch cannot override the MachinePolicy
# and UserPolicy scopes, and you do not need admin rights to work around it: pipe the file in
# instead. Text arriving on the pipeline is not a script file, so the file policy does not apply.
#
#   Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
#
# verify_setup.py is the real check and it runs everywhere. This script exists for the case that
# breaks first on a locked-down Windows laptop: Python not being there at all, so the real check
# cannot start. It needs no administrator rights and installs nothing.
#
# UNVERIFIED: this script has not been executed on a real Windows laptop. Everything below is
# written from the documented behaviour of the cmdlets, not from a run. What to test on a real
# Windows image, in this order: (1) that it runs at all under a Group-Policy execution policy,
# both invocations above; (2) that the localhost probes answer when a system proxy is configured —
# the proxy handling differs between Windows PowerShell 5.1 and PowerShell 7 and both branches are
# below; (3) that Ollama installed but missing from this window's PATH produces the "open a new
# window" line rather than NOT READY; (4) that the Microsoft Store python.exe alias stub is skipped
# and the py launcher is found.

$ErrorActionPreference = "Continue"
$problems = @()
$warnings = @()

function Report($label, $ok, $detail, $fix) {
    $mark = if ($ok) { "  ok  " } else { " FAIL " }
    $colour = if ($ok) { "Green" } else { "Red" }
    Write-Host $mark -ForegroundColor $colour -NoNewline
    Write-Host " $label" -NoNewline
    if ($detail) { Write-Host "  $detail" -ForegroundColor DarkGray } else { Write-Host "" }
    if (-not $ok -and $fix) { $script:problems += $fix }
}

# Same shape, but a miss is recorded as a warning and never turns the result into NOT READY.
function ReportWarn($label, $ok, $detail, $advice) {
    $mark = if ($ok) { "  ok  " } else { " warn " }
    $colour = if ($ok) { "Green" } else { "Yellow" }
    Write-Host $mark -ForegroundColor $colour -NoNewline
    Write-Host " $label" -NoNewline
    if ($detail) { Write-Host "  $detail" -ForegroundColor DarkGray } else { Write-Host "" }
    if (-not $ok -and $advice) { $script:warnings += $advice }
}

# Localhost must not go through the system proxy. Where a system proxy is set without a localhost
# bypass, an unguarded Invoke-RestMethod sends http://localhost:11434 to the proxy, it fails, and
# this script reports "Ollama is not running" on a machine where it is.
# PowerShell 7 has -NoProxy; Windows PowerShell 5.1 does not and needs the static property.
$localOnly = @{}
$originalProxy = [System.Net.WebRequest]::DefaultWebProxy
if ((Get-Command Invoke-RestMethod).Parameters.ContainsKey("NoProxy")) {
    $localOnly = @{ NoProxy = $true }
} else {
    [System.Net.WebRequest]::DefaultWebProxy = $null
}

Write-Host ""
Write-Host "RAG Training Day - Windows setup check" -ForegroundColor DarkGray
Write-Host "$([System.Environment]::OSVersion.VersionString)" -ForegroundColor DarkGray
Write-Host ""

# Python -----------------------------------------------------------------------------
# The Microsoft Store ships an app-execution alias called python.exe that opens the Store and
# prints nothing. It resolves through Get-Command, so the version has to be parsed rather than
# trusted, and the loop falls through to py when it is.
$python = $null
foreach ($candidate in @("python", "python3", "py")) {
    $found = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($found) {
        $version = & $candidate --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) { $python = $candidate; break }
        }
    }
}
Report "Python 3.10 or newer" ($null -ne $python) $(if ($python) { "$python - $(& $python --version 2>&1)" }) `
    "Install Python from python.org. Tick 'Add python.exe to PATH' on the first screen. No admin rights are needed if you choose 'Install for me only'."

if ($python -eq "py") {
    $warnings += "Only the py launcher works on this machine, so use 'py -m pip install -r requirements.txt' and 'py scripts\verify_setup.py' everywhere the instructions say python."
}

# git ---------------------------------------------------------------------------------
# You already have the repository if you are running this, so a missing git is information
# rather than a failure - it only matters for pulling later updates.
$git = Get-Command git -ErrorAction SilentlyContinue
ReportWarn "git installed" ($null -ne $git) $(if ($git) { $git.Source }) `
    "git is not on PATH. You do not need it: download https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip, extract it, and work in the extracted folder."

# Ollama -------------------------------------------------------------------------------
# The server probe runs whether or not the CLI is on PATH. OllamaSetup.exe adds its folder to the
# user PATH, but a PowerShell window opened before the install keeps the environment it inherited,
# so Get-Command comes back empty on a machine where Ollama is running perfectly well.
$ollama = Get-Command ollama -ErrorAction SilentlyContinue

$serverUp = $false
$serverVersion = ""
try {
    $serverVersion = (Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 10 @localOnly).version
    $serverUp = $true
} catch {
    $serverUp = $false
}

if ($serverUp -and $null -eq $ollama) {
    ReportWarn "Ollama installed" $false "the server answers, but the ollama command is not on this window's PATH" `
        "Ollama is installed and running. This PowerShell window was opened before the install, so it does not know about the ollama command yet - close it and open a new one. Nothing is broken."
} else {
    Report "Ollama installed" ($null -ne $ollama) $(if ($ollama) { $ollama.Source }) `
        "Download OllamaSetup.exe from ollama.com. It installs per-user under %LOCALAPPDATA% and does not ask for administrator rights."
}

if ($serverUp) {
    Report "Ollama is running" $true "version $serverVersion" ""
} else {
    Report "Ollama is running" $false "nothing answered on localhost:11434" `
        "Start it: run 'ollama serve' in a terminal, or launch Ollama from the Start menu. If you are certain Ollama is running and this line still says nothing answered, tell the trainer - the probe may be routed through a system proxy."
}

# Models -------------------------------------------------------------------------------
if ($serverUp) {
    $tags = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 30 @localOnly).models
    foreach ($wanted in @("qwen2.5:3b", "bge-m3", "nomic-embed-text")) {
        $model = $tags | Where-Object { $_.name -eq $wanted -or $_.name -eq "$wanted`:latest" } | Select-Object -First 1
        $size = if ($model) { "{0:N1} GB" -f ($model.size / 1e9) } else { "" }
        Report "model $wanted" ($null -ne $model) $size "Run: ollama pull $wanted"
    }

    # helios-q2 is a fine-tune the trainer builds and hands out. It is on no registry, so
    # 'ollama pull' cannot find it, and only module 3 uses it - a miss is a warning. Nothing is
    # replayed in its place: notebook 02 has no recorded run, so the two probe cells print that
    # they are skipped and the cells that read the Q2 and Q3 fare sheets off disk still run.
    $finetune = $tags | Where-Object { $_.name -eq "helios-q2" -or $_.name -eq "helios-q2:latest" } | Select-Object -First 1
    ReportWarn "model helios-q2" ($null -ne $finetune) `
        $(if ($finetune) { "{0:N1} GB" -f ($finetune.size / 1e9) } else { "not installed - module 3's two probes will be skipped, the corpus diff still runs" }) `
        "helios-q2 is not on any registry, so 'ollama pull' will not find it. The trainer hands it out on a USB stick before the day; ask for it if you want module 3's two probe cells to run live. Without it they print (skipped - helios-q2 not installed) and the rest of the module, which reads the two fare sheets off disk, runs exactly as it would otherwise."
}

# Seeded offline assets ------------------------------------------------------------------
# Two things the notebooks read that are neither models nor packages: MNIST for module 2
# (11.6 MB) and the all-MiniLM-L6-v2 ONNX archive chromadb fetches the first time a collection
# embeds text - module 6's bake-off and module 8's notebook both hit it (83 MB).
# scripts\seed_offline_assets.py downloads both at home.
$minilm = Join-Path $env:USERPROFILE ".cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx\model.onnx"
ReportWarn "all-MiniLM-L6-v2 is cached" (Test-Path $minilm) `
    $(if (Test-Path $minilm) { $minilm } else { "modules 6 and 8 would download 83 MB on the day" }) `
    "Once Python and the packages are in place, run 'python scripts\seed_offline_assets.py' at home. It fetches MNIST and chromadb's default embedder, and scripts\verify_setup.py checks both properly."

# $PSScriptRoot is empty when this file is piped in rather than run as a file, and the piped
# invocation is run from the repository root, so fall back to the current directory.
$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
$mnist = Join-Path $repoRoot "notebooks\mnist_data\train-images-idx3-ubyte.gz"
ReportWarn "MNIST is on disk" (Test-Path $mnist) `
    $(if (Test-Path $mnist) { "notebooks\mnist_data" } else { "module 2 would download 11.6 MB on the day" }) `
    "Run 'python scripts\seed_offline_assets.py' at home - it fetches MNIST as well."

# VS Code ------------------------------------------------------------------------------
# The notebooks are percent-format .py files you run block by block in VS Code. Best-effort look
# in the usual places: if you have it and this says warn, ignore the line.
$vscode = (Get-Command code -ErrorAction SilentlyContinue) -ne $null `
    -or (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\Code.exe")) `
    -or (Test-Path (Join-Path $env:ProgramFiles "Microsoft VS Code\Code.exe"))
ReportWarn "VS Code" $vscode $(if ($vscode) { "found" } else { "could not find it in the usual places" }) `
    "The notebooks are .py files you run block by block in VS Code. Get the User Installer from code.visualstudio.com - it installs into your own profile and asks for no admin rights - then add the Python extension by Microsoft."

# Reachability, so an unreachable registry is diagnosed here rather than on the day ----
# This one is a real internet call, so the system proxy goes back on first, and Windows
# PowerShell 5.1 needs TLS 1.2 asked for by name on older builds.
if (-not $serverUp -or $problems.Count -gt 0) {
    [System.Net.WebRequest]::DefaultWebProxy = $originalProxy
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
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
    foreach ($w in $warnings) { Write-Host "  warn: $w" -ForegroundColor Yellow }
    Write-Host ""
    # Same arithmetic as DOWNLOAD_TOTAL in verify_setup.py: 1.9 + 1.2 + 0.274 GB of models,
    # 0.400 GB of installed packages, 0.083 GB of ONNX and 0.012 GB of MNIST = 3.869 GB.
    Write-Host "The pre-work puts about 3.9 GB on disk in total." -ForegroundColor DarkGray
    Write-Host "Do it at home, not on a connection twenty other people are sharing." -ForegroundColor DarkGray
    Write-Host "Model files live in %USERPROFILE%\.ollama\models if you need them copied from a stick." -ForegroundColor DarkGray
    Write-Host "Quit Ollama on both machines first. Merge the contents of the blobs and manifests" -ForegroundColor DarkGray
    Write-Host "folders rather than replacing them - replacing unregisters models you already had." -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

foreach ($w in $warnings) { Write-Host "  warn: $w" -ForegroundColor Yellow }
if ($warnings.Count -gt 0) { Write-Host "" }
Write-Host "Basic checks passed. Now run the full one:" -ForegroundColor Green
if ($python) { Write-Host "  $python scripts\verify_setup.py" }
else { Write-Host "  python scripts\verify_setup.py   (install Python first - see above)" }
Write-Host ""
