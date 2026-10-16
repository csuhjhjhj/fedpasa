param()
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$figures=Join-Path $projectRoot 'paper_www\figures'
$evidence=Join-Path $figures 'paper_evidence_editable_20261010.pptx'
$framework=Join-Path $figures 'framework_editable.pptx'
$privateRoot=Join-Path $projectRoot '.build_www\complete_figure_set'
[void][System.IO.Directory]::CreateDirectory($privateRoot)
$candidate=Join-Path $privateRoot 'candidate.pptx'
$pptWasRunning=@(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app=New-Object -ComObject PowerPoint.Application
$combined=$null
$sourceEvidence=$null
$sourceFramework=$null
try {
 $sourceEvidence=$app.Presentations.Open($evidence,-1,0,0)
 $sourceFramework=$app.Presentations.Open($framework,-1,0,0)
 $combined=$app.Presentations.Add(0)
 $combined.PageSetup.SlideWidth=1350
 $combined.PageSetup.SlideHeight=786
 # PowerPoint inserts each source slide with uniform scale and centered margins.
 # Verify that transform numerically before saving all native objects.
 $specifications=@(@($framework,$sourceFramework,1),@($evidence,$sourceEvidence,6),@($evidence,$sourceEvidence,7),@($evidence,$sourceEvidence,4),@($evidence,$sourceEvidence,5))
 $maxGeometryError=0.0
 foreach($spec in $specifications){
  $path=[string]$spec[0]
  $source=$spec[1]
  $number=[int]$spec[2]
  [void]$combined.Slides.InsertFromFile($path,$combined.Slides.Count,$number,$number)
  $srcSlide=$source.Slides.Item($number)
  $dstSlide=$combined.Slides.Item($combined.Slides.Count)
  if($srcSlide.Shapes.Count -ne $dstSlide.Shapes.Count){throw 'Native shape count changed during slide insertion.'}
  $factor=[Math]::Min(1350.0/$source.PageSetup.SlideWidth,786.0/$source.PageSetup.SlideHeight)
  $offsetX=(1350.0-$source.PageSetup.SlideWidth*$factor)/2
  $offsetY=(786.0-$source.PageSetup.SlideHeight*$factor)/2
  for($j=1;$j -le $srcSlide.Shapes.Count;$j++){
   $ss=$srcSlide.Shapes.Item($j)
   $ds=$dstSlide.Shapes.Item($j)
   foreach($delta in @([Math]::Abs($ds.Left-($offsetX+$ss.Left*$factor)),[Math]::Abs($ds.Top-($offsetY+$ss.Top*$factor)),[Math]::Abs($ds.Width-$ss.Width*$factor),[Math]::Abs($ds.Height-$ss.Height*$factor))){$maxGeometryError=[Math]::Max($maxGeometryError,$delta)}
  }
 }
 if($maxGeometryError -gt 1.0){throw ('Nonuniform insertion detected: '+$maxGeometryError+' pt')}
 $combined.SaveAs($candidate,24)
 Write-Output ('Merged '+$combined.Slides.Count+' slides. Maximum fit-transform error: '+$maxGeometryError+' pt.')
 Write-Output $candidate
}
finally {
 if($null -ne $combined){$combined.Close()}
 if($null -ne $sourceEvidence){$sourceEvidence.Close()}
 if($null -ne $sourceFramework){$sourceFramework.Close()}
 if(-not $pptWasRunning -and $app.Presentations.Count -eq 0){$app.Quit()}
 [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
}

# Keep numeric caches identical to the preserved chart workbook cells.
Push-Location -LiteralPath $projectRoot
try {
@'
from pathlib import Path
import zipfile,xml.etree.ElementTree as E,io,posixpath,re
p=Path('.build_www/complete_figure_set/candidate.pptx')
c={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
x={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(p) as z:data={n:z.read(n) for n in z.namelist()}
count=0;maxdiff=0
for name in list(data):
 if not re.fullmatch(r'ppt/charts/chart\d+\.xml',name):continue
 root=E.fromstring(data[name]);relname=posixpath.dirname(name)+'/_rels/'+posixpath.basename(name)+'.rels'
 rels={r.attrib['Id']:r.attrib['Target'] for r in E.fromstring(data[relname])}
 external=root.find('c:externalData',c)
 if external is None:continue
 target=posixpath.normpath(posixpath.join(posixpath.dirname(name),rels[external.attrib['{'+c['r']+'}id']]))
 with zipfile.ZipFile(io.BytesIO(data[target])) as wb:
  wroot=E.fromstring(wb.read('xl/workbook.xml'));wrels={r.attrib['Id']:r.attrib['Target'] for r in E.fromstring(wb.read('xl/_rels/workbook.xml.rels'))}
  shared=[]
  if 'xl/sharedStrings.xml' in wb.namelist():shared=[''.join(e.itertext()) for e in E.fromstring(wb.read('xl/sharedStrings.xml'))]
  sheets={}
  for sheet in wroot.findall('x:sheets/x:sheet',x):
   rel=sheet.attrib['{'+c['r']+'}id'];sheetpath=posixpath.normpath(posixpath.join('xl',wrels[rel])).lstrip('/')
   cells={}
   for cell in E.fromstring(wb.read(sheetpath)).findall('.//x:c',x):
    v=cell.find('x:v',x)
    if cell.get('t')=='inlineStr':value=''.join(t.text or '' for t in cell.findall('.//x:t',x))
    elif v is None:continue
    elif cell.get('t')=='s':value=shared[int(v.text)]
    else:value=v.text
    cells[cell.get('r')]=value
   sheets[sheet.get('name')]=cells
  for typ in ['numRef','strRef']:
   for ref in root.findall('.//c:'+typ,c):
    formula=ref.find('c:f',c).text
    m=re.fullmatch(r"'?(.+?)'?!(\$?[A-Z]+\$?\d+)(?::(\$?[A-Z]+\$?\d+))?",formula)
    if not m:raise ValueError(formula)
    sheet,a,b=m.groups();a=a.replace('$','');b=(b or a).replace('$','');ma=re.fullmatch(r'([A-Z]+)(\d+)',a);mb=re.fullmatch(r'([A-Z]+)(\d+)',b)
    if ma[1]!=mb[1]:raise ValueError('Expected column range: '+formula)
    values=[sheets[sheet][ma[1]+str(i)] for i in range(int(ma[2]),int(mb[2])+1)]
    cache=ref.find('c:'+('numCache' if typ=='numRef' else 'strCache'),c)
    for pt in cache.findall('c:pt',c):
     value=values[int(pt.get('idx'))];old=pt.find('c:v',c).text
     if typ=='numRef':
      diff=abs(float(value)-float(old));maxdiff=max(maxdiff,diff)
      if diff>1e-10:raise ValueError('Substantive data discrepancy: '+name+' '+formula)
     if old!=value:pt.find('c:v',c).text=value;count+=1
 data[name]=E.tostring(root,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
print('Refreshed',count,'cache strings from existing workbook cells; maximum numeric difference',maxdiff)

'@ | & 'C:\Users\hejie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -
if($LASTEXITCODE -ne 0){throw 'Chart cache synchronization failed.'}
@'
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs/promises';
const workspaceDir=process.cwd();
const deps='C:/Users/hejie/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill='C:/Users/hejie/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
process.env.NODE_PATH=deps+'/node/node_modules';process.env.RUNTIME_NODE_MODULES=deps+'/node/node_modules';
process.env.RUNTIME_NODE=deps+'/node/bin/node.exe';process.env.RUNTIME_PYTHON=deps+'/python/python.exe';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=path.join(workspaceDir,'.build_www','complete_figure_set');
const candidatePath=path.join(build,'candidate.pptx');
await fs.mkdir(path.join(build,'validated'),{recursive:true});
const finalPath=path.join(build,'validated',`validated_${Date.now()}.pptx`);
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','17145000,9982200','--validate-heading-fit','--require-native-table-slide','5'],
 explicitTotalSlideCount:5,requiredNativeChartOwnerSlides:[2,3,4],requiredNativeTableOwnerSlides:[5],
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation_${Date.now()}.json`)});
const output=path.join(workspaceDir,'paper_www','figures','FedPASA_figures_complete_20261010.pptx');
await fs.copyFile(finalPath,output);console.log(output);

'@ | & 'C:\Users\hejie\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' --input-type=module
if($LASTEXITCODE -ne 0){throw 'Figure deck validation failed.'}
} finally { Pop-Location }
