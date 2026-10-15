# Requires a TeX distribution with pdflatex and bibtex. Run: .\build.ps1
$ErrorActionPreference = 'Stop'
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('FedPASA_latex_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir | Out-Null
Push-Location -LiteralPath $PSScriptRoot
try {
    & pdflatex '-interaction=batchmode' '-halt-on-error' "-output-directory=$buildDir" 'main.tex' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "LaTeX failed: $buildDir\main.log" }
    Copy-Item -LiteralPath 'references.bib','ACM-Reference-Format.bst' -Destination $buildDir
    Push-Location -LiteralPath $buildDir
    try {
        & bibtex 'main' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed: $buildDir\main.blg" }
    }
    finally { Pop-Location }
    foreach ($pass in 1..2) {
        & pdflatex '-interaction=batchmode' '-halt-on-error' "-output-directory=$buildDir" 'main.tex' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed: $buildDir\main.log" }
    }
    Copy-Item -LiteralPath (Join-Path $buildDir 'main.pdf') -Destination (Join-Path $PSScriptRoot 'main.pdf') -Force
    Select-String -LiteralPath (Join-Path $buildDir 'main.log') -Pattern 'Output written|Overfull|undefined|multiply defined' | ForEach-Object { $_.Line }
}
finally { Pop-Location }
