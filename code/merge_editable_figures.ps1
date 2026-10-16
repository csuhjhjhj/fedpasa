param()
$ErrorActionPreference='Stop'
$paper=Join-Path (Split-Path $PSScriptRoot -Parent) 'paper_www'
$figures=Join-Path $paper 'figures'
$exp=Join-Path $figures 'paper_evidence_editable.pptx'
$framework=Join-Path $figures 'framework_editable.pptx'
$pptWasRunning=@(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app=New-Object -ComObject PowerPoint.Application
$pres=$null
try {
 $pres=$app.Presentations.Add(0)
 $pres.PageSetup.SlideWidth=1350
 $pres.PageSetup.SlideHeight=786
 [void]$pres.Slides.InsertFromFile($exp,0,1,1)
 [void]$pres.Slides.InsertFromFile($framework,1,1,1)
 [void]$pres.Slides.InsertFromFile($exp,2,5,6)
 [void]$pres.Slides.InsertFromFile($exp,4,3,3)
 [void]$pres.Slides.InsertFromFile($exp,5,7,7)
 [void]$pres.Slides.InsertFromFile($exp,6,4,4)
 $pres.SaveAs((Join-Path $figures 'FedPASA_figures_editable.pptx'),24)
 $pres.Slides.Item(3).Export((Join-Path $paper '.build_www/merged-evidence-preview.png'),'PNG',2112,1230)
 $pres.Slides.Item(2).Export((Join-Path $paper '.build_www/merged-framework-preview.png'),'PNG',2112,1230)
 Write-Output ('Merged '+$pres.Slides.Count+' editable figure slides.')
}
finally {
 if($null -ne $pres){$pres.Close();[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pres)}
 if(-not $pptWasRunning -and $app.Presentations.Count -eq 0){$app.Quit()}
 [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
}