param([string]$DeckName = 'paper_evidence_editable_20261010.pptx')
$ErrorActionPreference = 'Stop'
$figures = Join-Path (Split-Path $PSScriptRoot -Parent) 'paper_www\figures'
$names = @('motivation_ppt','source_gate_ppt','buckets_ppt','sensitivity_ppt','cross_results_ppt','ablation_ppt','robustness_ppt','external_baselines_ppt','joint_evidence_ppt')
$alreadyRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open((Join-Path $figures $DeckName), -1, 0, 0)
    if ($deck.Slides.Count -ne $names.Count) { throw 'Unexpected slide count.' }
    for ($i=1; $i -le $names.Count; $i++) {
        $deck.Slides.Item($i).Export((Join-Path $figures ($names[$i-1]+'.png')), 'PNG', 4224, 1800)
    }
    Write-Output 'Exported 9 PNG figures at 4224 x 1800.'
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if (-not $alreadyRunning -and $app.Presentations.Count -eq 0) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
}
