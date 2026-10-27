[CmdletBinding()]
param(
    [string]$Model = 'deepseek-flash'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$workspacePackages = Join-Path $taskRoot 'work\python-deps'
$savedEnvironment = @{}
foreach ($name in @('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL', 'PYTHONPATH', 'PYTHONIOENCODING')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

$keyPointer = [IntPtr]::Zero
$secureKey = $null
$runExitCode = 1
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath $venvPython) {
        $pythonExecutable = $venvPython
    } elseif ((Test-Path -LiteralPath $bundledPython) -and (Test-Path -LiteralPath $workspacePackages)) {
        $pythonExecutable = $bundledPython
        $env:PYTHONPATH = if ($env:PYTHONPATH) { "$workspacePackages;$env:PYTHONPATH" } else { $workspacePackages }
    } else {
        $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
    }

    & $pythonExecutable -c 'import pandas, numpy, sklearn, matplotlib, threadpoolctl'
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependencies are missing. First create .venv and install requirements.lock.txt as described in README.'
    }
    $env:PYTHONIOENCODING = 'utf-8'
    $env:LLM_BASE_URL = 'https://api.deepseek.com'
    $env:LLM_MODEL = $Model
    Write-Host 'Enter your NEW DeepSeek API key below. Input is hidden; do not send the key in chat.'
    $secureKey = Read-Host 'API key' -AsSecureString
    if ($secureKey.Length -eq 0) { throw 'The API key is empty.' }
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:LLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $keyPointer = [IntPtr]::Zero

    Write-Host "Running real model experiment with $Model. No API key will be written to project files."
    $runArguments = @('run.py', '--agent', 'required')
    & $pythonExecutable @runArguments
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -eq 0) {
        Write-Host 'Run completed. Copy the final output path, not the API key, back to Codex for review.'
    } else {
        Write-Host "Run did not complete the real model group (exit $runExitCode). Share the error/status only."
    }
} finally {
    if ($keyPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer) }
    if ($secureKey) { $secureKey.Dispose() }
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Pop-Location
}
exit $runExitCode
