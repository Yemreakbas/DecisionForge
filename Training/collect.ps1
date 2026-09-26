<#
.SYNOPSIS
  Runs the headless data collector: several instances side by side, one seed each.

.DESCRIPTION
  Build the collector first, in Unity: JEV > Build Data Collector.
  Each instance plays its own Utility-vs-Utility match with exploration on and
  writes a run folder under -Out, which inspect_dataset.py and the phase 3
  trainer read. Unity can stay open meanwhile: the collector is a separate player.

.EXAMPLE
  .\Training\collect.ps1 -Instances 4 -Rounds 500
#>
param(
    [int]$Instances = 4,
    [int]$Rounds = 500,
    [int]$FirstSeed = 1,
    [ValidateSet("duel", "squad")][string]$Format = "duel",
    [double]$Explore = 0.012,
    [string]$Out = (Join-Path $PSScriptRoot "..\Datasets"),
    # Collect other arenas into a subfolder (e.g. -Out Datasets\warehouse): find_runs
    # only looks one level down, so the top level stays the default training set.
    [ValidateSet("octagon", "warehouse", "divide", "procedural")][string]$Arena = "octagon",
    # Procedural only: instance i builds layout FirstLayoutSeed + i, so one call
    # collects as many different arenas as it has instances.
    [int]$FirstLayoutSeed = 1
)

$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "..\Builds\Collector\JevCollector.exe"
if (-not (Test-Path $exe)) { throw "Collector not found at $exe. In Unity: JEV > Build Data Collector." }

$Out = (New-Item -ItemType Directory -Force -Path $Out).FullName

# The player parses numbers with the invariant culture; this machine's locale
# would otherwise hand it "0,012".
$exploreText = $Explore.ToString([System.Globalization.CultureInfo]::InvariantCulture)

$processes = @()
for ($i = 0; $i -lt $Instances; $i++) {
    $seed = $FirstSeed + $i
    $log = Join-Path $Out "collector_seed$seed.log"

    # One quoted string rather than an array: Windows PowerShell passes array
    # elements unquoted, and this project lives under a path with a space in it.
    $arguments = "-batchmode -nographics -collect -rounds $Rounds -seed $seed -format $Format " +
                 "-arena $Arena -explore $exploreText -out `"$Out`" -logFile `"$log`""
    if ($Arena -eq "procedural") { $arguments += " -layoutseed $($FirstLayoutSeed + $i)" }

    $process = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru -WindowStyle Hidden
    # Touching the handle now is what makes ExitCode readable once the process ends.
    $null = $process.Handle
    $processes += $process
}

Write-Host "Started $Instances collector(s), seeds $FirstSeed..$($FirstSeed + $Instances - 1), $Rounds rounds each -> $Out"
$processes | Wait-Process

# A collector that hits an exception quits with code 1 (MatchDirector.QuitOnException).
$failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
foreach ($p in $failed) { Write-Warning "Collector PID $($p.Id) exited with code $($p.ExitCode); its log is in $Out" }

Write-Host "Done. Check a run with: Training\.venv\Scripts\python Training\inspect_dataset.py <run folder>"
if ($failed.Count -gt 0) { exit 1 }
