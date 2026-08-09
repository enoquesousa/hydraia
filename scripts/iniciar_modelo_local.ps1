param(
    [ValidateRange(0.0, 1.0)]
    [double]$GpuOffload = 0.40,

    [ValidateRange(1024, 262144)]
    [int]$ContextLength = 16384
)

$ErrorActionPreference = "Stop"

$lmsCommand = Get-Command lms -ErrorAction SilentlyContinue
if ($lmsCommand) {
    $lms = $lmsCommand.Source
} else {
    $lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
}

if (-not (Test-Path -LiteralPath $lms)) {
    throw "LM Studio CLI nao encontrado. Instale e abra o LM Studio ao menos uma vez."
}

# `lms ps` usa stderr para informar que nao ha modelos carregados. Isso e um
# estado valido de inicializacao, nao uma falha do script.
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$loaded = (& $lms ps 2>&1 | Out-String)
$ErrorActionPreference = $previousErrorAction
if ($loaded -notmatch "hydraia-synth") {
    & $lms load qwen/qwen3.5-35b-a3b `
        --gpu $GpuOffload `
        --context-length $ContextLength `
        --parallel 1 `
        --identifier hydraia-synth `
        --no-speculative-draft-mtp `
        --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao carregar o modelo local."
    }
}

$serverRunning = Test-NetConnection `
    -ComputerName "127.0.0.1" `
    -Port 1234 `
    -InformationLevel Quiet `
    -WarningAction SilentlyContinue
if (-not $serverRunning) {
    & $lms server start --port 1234
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao iniciar o servidor local."
    }
}

& $lms ps

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi `
        --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu `
        --format=csv,noheader
}
