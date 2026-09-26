<#
.SYNOPSIS
  Headless A/B evaluation: one brain against another, several matches side by side.

.DESCRIPTION
  Build the player first, in Unity: JEV > Build Data Collector (the same build
  collects data and evaluates). Each instance plays one match with -evaluate:
  fixed 60 Hz steps as fast as the machine allows, no recording, CSV into
  Builds\Collector\MatchLogs tagged _eval and with its seed. Half of the
  instances put the brain under test on team A, half on team B -- spawns are
  already swapped every round inside a match.

  -nographics means no GPU, so a JEV brain runs on the CPU backend here.

.EXAMPLE
  .\Training\evaluate.ps1 -Instances 4 -Rounds 100
#>
param(
    [int]$Instances = 4,
    [int]$Rounds = 100,
    [int]$FirstSeed = 101,
    [ValidateSet("duel", "squad")][string]$Format = "duel",
    [ValidateSet("jev", "utility", "stopshoot")][string]$Brain = "jev",
    [ValidateSet("jev", "utility", "stopshoot")][string]$Against = "utility",
    # Rules ablation for both teams: Weapon.MovingSpreadPenalty. Negative = real rules.
    [double]$SpreadPenalty = -1,
    # Octagon: where the baseline was tuned. Warehouse, Divide: phase 5 arenas.
    [ValidateSet("octagon", "warehouse", "divide")][string]$Arena = "octagon",
    # Checkpoint name of the world model a JEV team plays with (MatchDirector.AlternativeJevModels).
    [string]$JevModel = ""
)

$ErrorActionPreference = "Stop"

$root = Join-Path $PSScriptRoot ".."
$exe = Join-Path $root "Builds\Collector\JevCollector.exe"
if (-not (Test-Path $exe)) { throw "Player not found at $exe. In Unity: JEV > Build Data Collector." }

$logs = (New-Item -ItemType Directory -Force -Path (Join-Path $root "Builds\Collector\EvalLogs")).FullName
$started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

$processes = @()
for ($i = 0; $i -lt $Instances; $i++) {
    $seed = $FirstSeed + $i
    $a, $b = if ($i % 2 -eq 0) { $Brain, $Against } else { $Against, $Brain }
    $log = Join-Path $logs "eval_seed$seed.log"

    # One quoted string: this project lives under a path with a space in it.
    $arguments = "-batchmode -nographics -evaluate -brainA $a -brainB $b -rounds $Rounds -seed $seed " +
                 "-format $Format -arena $Arena -logFile `"$log`""
    if ($JevModel -ne "") { $arguments += " -jevmodel $JevModel" }
    if ($SpreadPenalty -ge 0) {
        # Invariant culture: this machine's locale would hand the player "0,5".
        $arguments += " -spreadpenalty " + $SpreadPenalty.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    }

    $process = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru -WindowStyle Hidden
    $null = $process.Handle
    $processes += $process
}

$rules = if ($SpreadPenalty -ge 0) { ", ABLATION moving spread penalty = $SpreadPenalty" } else { "" }
if ($JevModel -ne "") { $rules += ", JEV model $JevModel" }
Write-Host "Started $Instances match(es) in $Arena`: $Brain vs $Against, $Rounds rounds each, seeds $FirstSeed..$($FirstSeed + $Instances - 1)$rules"
$processes | Wait-Process

$failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
foreach ($p in $failed) { Write-Warning "Match PID $($p.Id) exited with code $($p.ExitCode); logs in $logs" }

& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "summarize_matches.py") --since ($started - 1)
if ($failed.Count -gt 0) { exit 1 }
