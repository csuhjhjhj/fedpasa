/** Editable companions to the five existing paper plots. Reads existing results only. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.dirname(here);
const build = path.join(workspaceDir, '.build_www', 'experiment_pptx');
const runtime = 'C:/Users/hejie/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill = 'C:/Users/hejie/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const python = runtime + '/python/python.exe';
process.env.NODE_PATH = runtime + '/node/node_modules';
process.env.RUNTIME_NODE_MODULES = runtime + '/node/node_modules';
process.env.RUNTIME_NODE = runtime + '/node/bin/node.exe';
process.env.RUNTIME_PYTHON = python;
const { Presentation, PresentationFile } = await import(pathToFileURL(runtime + '/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(skill + '/container_tools/artifact_tool_utils.mjs').href);
await fs.mkdir(build, { recursive: true });
const presentation = Presentation.create({ slideSize: { width: 1408, height: 820 } });
const font = 'Arial';
const colors = { coral: '#DF715D', blue: '#487EB5', mint: '#56A68C', purple: '#8C77B4', gold: '#CBA24C', ink: '#25344B', gray: '#8C95A4', grid: '#DEE4ED' };
const textStyle = { typeface: font, fontSize: 24, fill: colors.ink };
function text(slide, value, x, y, w, h, size=26, bold=false, align='center', color=colors.ink) {
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value; shape.text.style={typeface:font,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none'}; return shape;
}
function title(slide,value){text(slide,value,48,24,1312,52,34,true);}
function line(slide,x1,y1,x2,y2,color=colors.ink,width=1.5,dashed=false){
  return slide.shapes.add({geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1)||0.001,height:Math.abs(y2-y1)||0.001,verticalFlip:y2<y1},fill:'none',line:{fill:color,width,style:dashed?'dashed':'solid'}});
}
function axis(titleValue,limits={}){return {visible:true,...(titleValue?{title:{text:titleValue,textStyle:{...textStyle,fontSize:25}}}:{}),textStyle:{...textStyle,fontSize:21},line:{fill:colors.ink,width:1.4},majorGridlines:{fill:colors.grid,width:0.8},...limits};}
function chart(slide,type,config){
  // Excel stores 15 significant digits. Preserve all source points at that
  // precision so PowerPoint's embedded workbook and chart cache stay identical.
  for(const s of config.series??[])for(const key of ['values','xValues'])if(s[key])s[key]=s[key].map(v=>Number(v.toPrecision(15)));
  const c=slide.charts.add(type,{chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},hasLegend:false,...config});applyPresentationChartFont(c,{fontFamily:font});return c;
}
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const readJson=async f=>JSON.parse((await fs.readFile(f,'utf8')).replace(/^\uFEFF/,''));
const out=path.join(here,'outputs','www_v1');

// 1: Conceptual motivation, with the same 200 samples and formulas as the source.
{
 const slide=presentation.slides.add(); slide.background.fill='#FFFFFF';
 const freq=Array.from({length:200},(_,i)=>10**(3.3*i/199));
 const q=freq.map(f=>1/Math.sqrt(f+1)), qmin=Math.min(...q), qmax=Math.max(...q);
 const aware=q.map(v=>0.8*(1-0.75*((v-qmin)/(qmax-qmin))**0.7));
 const eta=Array.from({length:200},(_,i)=>0.2+2*i/199), rho=eta.map(v=>0.2*Math.max(0,Math.min(1,(1.5-v)/0.5)));
 title(slide,'Evidence-aware adaptation');
 chart(slide,'scatter',{position:{left:48,top:126,width:650,height:606},title:'(a) Item-level routing',titleTextStyle:{...textStyle,fontSize:28,bold:true},scatterOptions:{style:'line'},
  series:[{name:'Uniform adaptation',xValues:freq,values:freq.map(()=>0.8),line:{fill:colors.gray,width:3,style:'dashed'},marker:{symbol:'none'}},{name:'FedPASA (evidence-aware)',xValues:freq,values:aware,line:{fill:colors.coral,width:4},marker:{symbol:'none'}}],
  xAxis:axis('Item interaction frequency (log scale)',{min:1,max:2200,numberFormatCode:'0',tickLabelPosition:'low'}),yAxis:axis('Behavioral adaptation strength',{min:0,max:0.9,majorUnit:0.2,numberFormatCode:'0.0'}),hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{...textStyle,fontSize:22}}});
 chart(slide,'scatter',{position:{left:728,top:126,width:632,height:606},title:'(b) Density-gated anchoring',titleTextStyle:{...textStyle,fontSize:28,bold:true},scatterOptions:{style:'line'},
  series:[{name:'No anchoring',xValues:eta,values:eta.map(()=>0),line:{fill:colors.gray,width:3,style:'dashed'},marker:{symbol:'none'}},{name:'AutoAnchor ρeff',xValues:eta,values:rho,line:{fill:colors.mint,width:4},marker:{symbol:'none'}}],
  xAxis:axis('Target density ηT = |UT| / |IT|',{min:0,max:2.2,majorUnit:0.5,numberFormatCode:'0.0',tickLabelPosition:'low'}),yAxis:axis('Anchor weight',{min:0,max:0.22,majorUnit:0.05,numberFormatCode:'0.00'}),hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{...textStyle,fontSize:22}}});
 slide.speakerNotes.textFrame.setText('Conceptual functions from www_workspace/make_motivation_figure.py. These are illustrative mechanism curves, not measured experimental results. 200 original sampling points retained per curve.');
}

// 2: Source reliability, retaining the six existing values exactly.
{
 const data=await readJson(path.join(out,'source_gate.json'));
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF'; title(slide,'Source reliability gate');
 const pos={left:88,top:112,width:1232,height:652};
 const series=data.points.map(([name,x,y],i)=>({name:name.replace('_to_',' → '),xValues:[x],values:[y*100],fill:y>0?colors.mint:colors.coral,line:{fill:'none',width:0},marker:{symbol:'circle',size:12},dataLabelOverrides:[{idx:0,text:name.replace('_to_',' → '),position:i===4?'b':'t',showValue:false,showSeriesName:true,textStyle:{...textStyle,fontSize:21}}]}));
 series.push({name:'Zero source benefit',xValues:[0.191,0.215],values:[0,0],line:{fill:colors.gray,width:1.5},marker:{symbol:'none'}});
 series.push({name:'τ = 0.2025',xValues:[data.tau,data.tau],values:[-6.5,2],line:{fill:colors.blue,width:2.4,style:'dashed'},marker:{symbol:'none'}});
 chart(slide,'scatter',{position:pos,scatterOptions:{style:'lineWithMarkers'},series,
 xAxis:axis('Source adapter zero-shot HR@20 on target validation',{min:0.190,max:0.215,majorUnit:0.005,numberFormatCode:'0.000',tickLabelPosition:'low'}),
 yAxis:axis('Source benefit (src − no-src), ×10⁻²',{min:-6,max:2,majorUnit:2,numberFormatCode:'0.0'}),dataLabels:{showValue:false,showSeriesName:false,textStyle:{...textStyle,fontSize:21}}});
 text(slide,'Accept source\nτ = 0.2025',860,575,260,80,25,false,'left',colors.blue);
 slide.speakerNotes.textFrame.setText('Source: www_workspace/outputs/www_v1/source_gate.json. x is existing seed-42 zero-shot target-validation HR@20. y is existing five-seed mean source minus no-source HR@20, displayed ×100. Original six values preserved.');
}

// 3: True bucket aggregates over the existing six transfer directions and five seeds.
{
 const methods=['transfr','fedmoe','fedpasa'], buckets=['zero','1_5','6_20','20plus'];
 const all=Object.fromEntries(methods.map(m=>[m,Object.fromEntries(buckets.map(b=>[b,[]]))]));
 const dirs=await fs.readdir(path.join(out,'cross'),{withFileTypes:true});
 for(const dir of dirs.filter(d=>d.isDirectory()))for(const m of methods){const base=path.join(out,'cross',dir.name,m);let seeds=[];try{seeds=await fs.readdir(base,{withFileTypes:true});}catch{continue;}
  for(const seed of seeds.filter(d=>d.isDirectory())){let r;try{r=await readJson(path.join(base,seed.name,'results.json'));}catch{continue;}for(const b of buckets){const val=r.metrics_no_pap?.[`s99freq_${b}_HR@20`];if(typeof val==='number')all[m][b].push(val);}}
 }
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';title(slide,'Cross-domain HR@20 by target-item frequency');
 const series=methods.map((m,i)=>({name:['TransFR','FedMoE','FedPASA'][i],values:buckets.map(b=>{if(!all[m][b].length)throw new Error(`No existing result for ${m}/${b}`);return mean(all[m][b]);}),fill:[colors.purple,colors.blue,colors.coral][i],line:{fill:'none',width:0}}));
 chart(slide,'bar',{position:{left:116,top:124,width:1176,height:638},categories:['0','1–5','6–20','>20'],series,barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
 xAxis:axis('Target item frequency bucket',{majorGridlines:null}),yAxis:axis('HR@20',{min:0,max:0.65,majorUnit:0.1,numberFormatCode:'0.0'}),hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:26}}});
 slide.speakerNotes.textFrame.setText('Source: www_workspace/outputs/www_v1/cross/{direction}/{method}/seed*/results.json. Metric metrics_no_pap.s99freq_{zero,1_5,6_20,20plus}_HR@20, mean over available metric values in the same six-direction, five-seed runs, matching make_paper_figures.py and tables_buckets.tex. Means retain 15 significant digits in the embedded workbook.');
}

// 4: Existing sensitivity points only. Absent default settings are not invented.
{
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';title(slide,'Sensitivity on Music → Movies (three seeds)');text(slide,'HR@20',35,130,120,34,24);
 const sweeps=[['alpha','α (routing strength)',[0.25,0.4,0.55,0.7,0.85],colors.coral],['coldpower','κ (coldness power)',[0.5,0.7,1,1.5],colors.purple],['condscale','λ (conditioning)',[0,0.2,0.4,0.6],colors.blue],['anchorw','ρ (max anchor)',[0,0.1,0.2,0.3,0.5],colors.mint]];
 for(let i=0;i<sweeps.length;i++){const [name,label,values,color]=sweeps[i],xs=[],ys=[];
  for(const val of values){const spelling=Number.isInteger(val)?val.toFixed(1):String(val);let seeds=[];const base=path.join(out,'sensitivity',name+spelling);try{seeds=await fs.readdir(base,{withFileTypes:true});}catch{continue;}const pts=[];for(const seed of seeds.filter(d=>d.isDirectory())){const r=await readJson(path.join(base,seed.name,'results.json'));if(typeof r.metrics_no_pap?.['s99_HR@20']==='number')pts.push(r.metrics_no_pap['s99_HR@20']);}if(pts.length){if(pts.length!==3)throw new Error('Expected three seeds: '+base);xs.push(val);ys.push(mean(pts));}}
  chart(slide,'scatter',{position:{left:35+344*i,top:166,width:337,height:556},scatterOptions:{style:'lineWithMarkers'},series:[{name:'FedPASA',xValues:xs,values:ys,fill:color,line:{fill:color,width:3.5},marker:{symbol:'circle',size:9}}],
  xAxis:axis(label,{min:Math.min(...xs),max:Math.max(...xs),numberFormatCode:'0.00',tickLabelPosition:'low',textStyle:{...textStyle,fontSize:18},title:{text:label,textStyle:{...textStyle,fontSize:23}}}),yAxis:axis('',{min:0.23,max:0.35,majorUnit:0.02,numberFormatCode:'0.00',textStyle:{...textStyle,fontSize:19}})});
 }
 slide.speakerNotes.textFrame.setText('Source: existing www_workspace/outputs/www_v1/sensitivity/{setting}/seed{42,123,456}/results.json, metrics_no_pap.s99_HR@20. Same one-factor-at-a-time values as the source figure. Default points absent from the source directory were left absent, as in the original plotting code.');
}

// 5: Recover the original t-SNE scatter marker geometry without a new projection.
{
 const pdf=path.join(workspaceDir,'paper_www','figures','tsne_anchor.pdf');
 const extract="import fitz,json,sys\np=fitz.open(sys.argv[1])[0]\narr=[]\nfor d in p.get_drawings():\n r=d['rect'];fill=d.get('fill')\n if d['type']=='fs' and fill and abs(r.width-2)<.01 and abs(r.height-2)<.01:\n  arr.append({'rect':list(r),'circle':d['items'][0][0]=='c','rgb':list(fill),'opacity':d.get('fill_opacity',1),'width':d.get('width',1)})\nprint(json.dumps(arr))";
 const marks=JSON.parse(execFileSync('C:/Users/hejie/miniconda3/python.exe',['-c',extract,pdf],{encoding:'utf8',maxBuffer:10*1024*1024}));
 if(marks.length!==2042)throw new Error('Unexpected t-SNE marker count: '+marks.length);
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';title(slide,'Shared PCA → t-SNE projection');
 text(slide,'Same items before and after adaptation · Music → Movies, seed 42',88,91,1232,42,27);
 const k=2.6,ox=48,oy=206;
 for(const [i,m] of marks.entries()){const [x0,y0,x1,y1]=m.rect;const color=m.circle?colors.coral:colors.blue;
  slide.shapes.add({geometry:m.circle?'ellipse':'triangle',name:`tsne-${m.circle?'cold':'warm'}-${i+1}`,position:{left:ox+x0*k,top:oy+y0*k,width:(x1-x0)*k,height:(y1-y0)*k},fill:color+'/50',line:{fill:color+'/50',width:m.width*k,style:'solid'}});
 }
 const lefts=[4.25,170.8333282470703,337.4166564941406],w=162.3333282470703,top=31.060012817382812,h=130.2899932861328;
 for(let i=0;i<3;i++){const x=ox+lefts[i]*k,y=oy+top*k;line(slide,x,y,x+w*k,y);line(slide,x,y+h*k,x+w*k,y+h*k);line(slide,x,y,x,y+h*k);line(slide,x+w*k,y,x+w*k,y+h*k);text(slide,['Frozen prior xᵢ','TransFR (no anchor)','FedPASA (AutoAnchor)'][i],x,oy+12*k,w*k,46,26);}
 text(slide,'cold (≤5)',ox+115.28*k,oy+32*k,160,27,21,false,'left');
 text(slide,'warm (>20)',ox+115.28*k,oy+42.5*k,172,27,21,false,'left');
 slide.speakerNotes.textFrame.setText('Marker positions, point count, shapes, and transparency recovered directly from paper_www/figures/tsne_anchor.pdf using PyMuPDF. 2040 experimental points (840 cold, 1200 warm) plus two legend markers. No training, inference, PCA, or t-SNE was rerun. Coordinates preserve the original three-panel projection up to a common translation and scale. Each scatter marker is a native PowerPoint shape.');
}

const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
// The chart facade has no logarithmic-axis option. Set the first scatter x-axis
// to log10 in the editable OOXML chart, preserving its actual frequency values.
const patch=String.raw`import zipfile,sys,xml.etree.ElementTree as E,os
p=sys.argv[1]
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
q=lambda tag:'{'+ns[tag.split(':')[0]]+'}'+tag.split(':')[1]
with zipfile.ZipFile(p) as z: data={n:z.read(n) for n in z.namelist()}
colors={1:['8C95A4','DF715D'],2:['8C95A4','56A68C'],3:['DF715D','56A68C','DF715D','DF715D','DF715D','56A68C','8C95A4','487EB5'],5:['DF715D'],6:['8C77B4'],7:['487EB5'],8:['56A68C']}
for key in [n for n in data if '/charts/chart' in n and n.endswith('.xml')]:
 idx=int(key.rsplit('chart',1)[1].split('.')[0]);r=E.fromstring(data[key])
 for node in r.findall('.//c:smooth',ns):node.set('val','0')
 for node in r.findall('.//c:varyColors',ns):node.set('val','0')
 for node in r.findall('.//c:scatterStyle',ns):node.set('val','lineMarker')
 for j,ser in enumerate(r.findall('.//c:ser',ns)):
  smooth=ser.find('c:smooth',ns)
  if smooth is None:smooth=E.SubElement(ser,q('c:smooth'))
  smooth.set('val','0')
  for dp in ser.findall('c:dPt',ns):ser.remove(dp)
  mark=ser.find('c:marker',ns)
  if mark is not None and idx in colors:
   old=mark.find('c:spPr',ns)
   if old is not None:mark.remove(old)
   col=colors[idx][j];sp=E.SubElement(mark,q('c:spPr'));fill=E.SubElement(sp,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':col});ln=E.SubElement(sp,q('a:ln'),{'w':'12700'});fill=E.SubElement(ln,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':col})
  if idx==3 and j<6:
   for dl in ser.findall('c:dLbls/c:dLbl',ns):
    label=dl.find('c:dLblPos',ns)
    if label is None:
     label=E.Element(q('c:dLblPos'));first=next((k for k,n in enumerate(dl) if n.tag.rsplit('}',1)[1].startswith('show')),len(dl));dl.insert(first,label)
    label.set('val',['r','r','t','b','b','t'][j])
 if idx==1:
  axes=r.findall('.//c:valAx',ns);a=next((a for a in axes if a.find('c:axPos',ns) is not None and a.find('c:axPos',ns).get('val')=='b'),axes[0]);sc=a.find('c:scaling',ns);sc.insert(0,E.Element(q('c:logBase'),{'val':'10'}))
 data[key]=E.tostring(r,encoding='utf-8',xml_declaration=True)
tmp=p+'.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
os.replace(tmp,p)`;
execFileSync(python,['-c',patch,candidate]);
const finalPath=path.join(workspaceDir,'paper_www','figures','experiment_figures_editable.pptx');
const validatedPath=path.join(build,'validated',`experiment_figures_${Date.now()}.pptx`);
await fs.mkdir(path.dirname(validatedPath),{recursive:true});
await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath:validatedPath,pythonExecutable:python,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','13411200,7810500','--validate-heading-fit'],explicitTotalSlideCount:5,
 requiredNativeChartOwnerSlides:[1,2,3,4],requiredNativeTableOwnerSlides:[],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation_${Date.now()}.json`)});
await fs.copyFile(validatedPath,finalPath);
console.log(finalPath);

