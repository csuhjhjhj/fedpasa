/** FedPASA framework, editable native PowerPoint shapes. Run with bundled Node. */
import fs from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.ARTIFACT_TOOL_ROOT || 'C:/Users/hejie/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool';
const skill=process.env.SKILL_DIR || 'C:/Users/hejie/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
process.env.RUNTIME_NODE_MODULES ||= path.resolve(runtime,'..','..');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'dist/artifact_tool.mjs')).href);
const build=path.join(root,'.build_www','framework_pptx');
const final=path.join(root,'paper_www','figures','framework_editable.pptx');
await fs.mkdir(build,{recursive:true});
const W=1800,H=1048,p=Presentation.create({slideSize:{width:W,height:H}}),s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#253C52',muted:'#577087',blue:'#3985C9',blueFill:'#DEEBF7',coral:'#DA7474',coralFill:'#FFB9B9',green:'#548A62',greenFill:'#D5E7D4',gold:'#B98A26',goldFill:'#FFD965',lilac:'#AA78B4',lilacFill:'#F2CEEE',line:'#7190AA'};
let n=0;const bounds=[];const tokens=[];
function shp(geo,x,y,w,h,fill='none',edge='none',lw=1.5,name=''){
 const q=s.shapes.add({geometry:geo,name:name||`shape-${++n}`,position:{left:x,top:y,width:w,height:h},fill,line:{fill:edge,width:lw,style:'solid'}});
 bounds.push({name:q.name,x,y,w,h});return q;
}
function box(x,y,w,h,fill='#FFFFFF',edge=C.line,lw=2,r=12,name='') {const q=shp('roundRect',x,y,w,h,fill,edge,lw,name);q.borderRadius=r;return q;}
function text(t,x,y,w,h=38,size=26,bold=false,color=C.ink,align='center'){
 const q=shp('textbox',x,y,w,h,'none','none',0,`label-${++n}`);q.text=t;
 q.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return q;
}
function seg(x1,y1,x2,y2,col=C.ink,lw=2.2,dash=false){const q=shp('line',Math.min(x1,x2),Math.min(y1,y2),Math.max(.01,Math.abs(x2-x1)),Math.max(.01,Math.abs(y2-y1)),'none',col,lw);q.position={...q.position,horizontalFlip:x2<x1,verticalFlip:y2<y1};q.line={fill:col,width:lw,style:dash?'dashed':'solid'};}
function line(a,col=C.ink,lw=2.2,dash=false){for(let i=1;i<a.length;i++)seg(...a[i-1],...a[i],col,lw,dash);}
function arrow(a,col=C.ink,lw=2.2,dash=false){line(a,col,lw,dash);const v=a.at(-1),u=a.at(-2);const q=shp('triangle',v[0]-6,v[1]-6,12,12,col,'none',0);q.position={...q.position,rotation:Math.atan2(v[1]-u[1],v[0]-u[0])*180/Math.PI+90};}
function dot(x,y,c=C.blue,r=4){shp('ellipse',x-r,y-r,r*2,r*2,c,'none',0);}
function vec(x,y,w=120,h=18,cs=[C.blueFill,'#5B9AD5',C.lilacFill,C.coralFill,C.goldFill],owner=null){cs.forEach((c,i)=>shp('rect',x+i*w/cs.length,y,w/cs.length,h,c,C.line,.8));if(owner)tokens.push({x,y,w,h,owner});}
function doc(x,y,col=C.blue,scale=1){for(let i=2;i>=0;i--)box(x+i*5*scale,y-i*4*scale,30*scale,42*scale,i?'#EDF5FC':'#FFFFFF',col,1.6,2);for(let i=0;i<4;i++)seg(x+6*scale,y+(9+i*7)*scale,x+24*scale,y+(9+i*7)*scale,col,1.2);}
function snow(x,y,r=12){for(let j=0;j<3;j++){const a=j*Math.PI/3,dx=Math.cos(a),dy=Math.sin(a);seg(x-r*dx,y-r*dy,x+r*dx,y+r*dy,C.blue,2);for(const q of [-1,1])for(const z of [-1,1]){const bx=x+q*r*.62*dx,by=y+q*r*.62*dy;seg(bx,by,bx+q*4*dx-z*4*dy,by+q*4*dy+z*4*dx,C.blue,1.6);}}}
function user(x,y,c=C.blue){shp('ellipse',x+11,y,22,22,c,'none',0);shp('ellipse',x,y+24,44,31,c,'none',0);}
function server(x,y){for(let i=0;i<3;i++){box(x,y+i*22,57,17,'#5B9AD5',C.ink,1.4,4);dot(x+9,y+i*22+8,C.goldFill,2.5);for(let j=0;j<3;j++)dot(x+35+j*7,y+i*22+8,'#FFFFFF',2);}}
function laptop(x,y,w=64,h=52){box(x+5,y,w-10,h-10,'#A9CEE9',C.ink,1.4,4);box(x+10,y+5,w-20,h-20,'#F7FBFE',C.blue,1,1);shp('trapezoid',x,y+h-10,w,10,'#5B9AD5',C.ink,1.2);}
function lock(x,y){box(x+8,y,18,20,'none',C.ink,2,7);box(x,y+12,34,27,C.blueFill,C.ink,1.5,4);dot(x+17,y+24,C.ink,3);}
function bars(x,y,col=C.blue){[14,26,40,56].forEach((v,i)=>shp('rect',x+i*14,y+56-v,9,v,col,'none',0));}
function sliders(x,y,w=85,col=C.green){[0,16,32].forEach((dy,i)=>{seg(x,y+dy,x+w,y+dy,col,2.3);box(x+[20,57,36][i],y+dy-6,10,12,'#FFFFFF',col,1.5,2);});}
function adapter(x,y,w,title,sub,col,fill,owner){box(x,y,w,134,'#FFFFFF',col,2,10);text(title,x+8,y+10,w-16,33,27,true,col);text(sub,x+7,y+44,w-14,27,23,false,C.muted);for(let i=2;i>=0;i--)box(x+33+i*10,y+87-i*7,49,22,fill,col,1.3,3);arrow([[x+97,y+100],[x+118,y+100]],col,1.8);vec(x+128,y+90,w-144,20,[fill,col,fill],owner);}
function panel(x,w,title,fill,col){const owner={x:x+8,y:249,w:w-16,h:452};box(x,241,w,468,fill,col,2.3,16);text(title,x+20,254,w-40,43,31,true,col,'left');seg(x+18,301,x+w-18,301,col,1.2);return owner;}
// (a) Source pretraining and (b) the parameter-exchange loop.
box(20,18,510,137,'#FFF1F1',C.coral,2.2,16);text('(a) Source initialization',43,29,465,39,31,true,C.ink,'left');doc(50,90,C.coral,1.12);text('Interactions + frozen features',110,74,393,34,26,false,C.ink,'left');text('BPR pretraining of head adapter',110,112,393,30,24,false,C.muted,'left');
box(550,18,1230,137,'#FAF1FB',C.lilac,2.2,16);text('(b) Federated optimization',575,29,900,39,31,true,C.ink,'left');server(586,83);text('Server / FedAvg',666,77,285,36,28,true,C.ink,'left');text('Adapters + conditioner',666,115,303,29,24,false,C.muted,'left');
arrow([[1000,89],[1285,89]],C.blue,2.5,true);arrow([[1285,128],[1000,128]],C.blue,2.5,true);text('broadcast',1060,59,160,27,23,false,C.blue);text('upload updates',1045,130,200,23,21,false,C.blue);
laptop(1320,88);laptop(1370,93,51,46);text('Target clients',1436,77,314,36,28,true,C.ink,'left');text('Local BPR updates',1436,115,314,29,24,false,C.muted,'left');
// (c) Main pipeline. A dedicated lower rail carries the frozen prior.
box(20,177,1760,673,'#FFFFFF','#5B9AD5',2.3,16);text('(c) Target client',45,187,500,40,31,true,C.ink,'left');text('Shared trainable modules, private user state',1010,189,737,36,25,false,C.muted,'right');
const pg=panel(260,460,'01  Item / PGDR',C.blueFill,C.blue);
const fm=panel(755,480,'02  Client / FiLM','#E6F2E5',C.green);
const an=panel(1270,485,'03  Catalog / AutoAnchor','#FFF2C9',C.gold);
// Frozen item encoder and feature input.
text('Item text',48,292,174,38,28,true);doc(104,344,C.blue,1.25);arrow([[135,401],[135,434]],C.blue);
box(49,441,172,88,'#F8EFF8',C.lilac,2,10);snow(76,466);text('Frozen',95,450,110,32,27);text('encoder',66,486,141,30,27);arrow([[135,529],[135,555]],C.blue);vec(62,562,146,22,undefined,{x:44,y:285,w:185,h:322});text('Semantic prior',41,588,188,32,24,false,C.muted);
// PGDR: graphic representation of frequency, two trainable paths and one gate.
bars(291,318,C.blue);text('Training frequency',355,312,335,33,27,true);text('sets the item routing weight',355,347,335,29,23,false,C.muted);
adapter(282,396,196,'Head adapter','source initialization',C.coral,'#FFB9B9',pg);adapter(502,396,196,'Tail adapter','target initialization',C.green,'#B4D6B3',pg);
line([[208,573],[244,573],[244,383],[600,383]],C.blue);arrow([[380,383],[380,396]],C.blue);arrow([[600,383],[600,396]],C.blue);dot(380,383,C.blue,3);arrow([[380,530],[380,543],[421,543],[421,558]],C.blue);arrow([[600,530],[600,543],[561,543],[561,558]],C.blue);
box(363,558,252,63,'#FFFFFF',C.blue,2,11);text('Frequency mix',375,565,228,32,27,true,C.blue);vec(403,602,171,12,[C.coralFill,C.coralFill,C.blueFill,C.greenFill,C.greenFill],pg);text('Both adapters are trainable',288,627,406,29,24,false,C.muted);
// FiLM: a private profile drives a shared conditioner, then feature modulation.
box(781,318,428,92,'#FFFFFF',C.green,1.7,10);text('Private history profile',808,323,374,30,26,true,C.green);doc(806,365,C.green,.70);arrow([[853,380],[891,380]],C.green,1.8);vec(900,367,121,21,['#DBECD6','#8ABA8E','#B4D6B3','#6EAB7E'],fm);arrow([[1031,380],[1063,380]],C.green,1.8);sliders(1076,361,97,C.green);
text('Shared conditioner',809,430,371,31,25,false,C.green);arrow([[995,410],[995,430]],C.green);arrow([[995,464],[995,497]],C.green);
box(791,505,408,108,'#FFFFFF',C.green,2,10);text('FiLM: scale + shift',810,514,370,35,28,true,C.green);vec(826,565,132,22,[C.coralFill,C.blueFill,C.greenFill,C.goldFill],fm);arrow([[971,576],[1012,576]],C.green);vec(1029,565,132,22,[C.lilacFill,'#8BBA98','#5B9AD5',C.goldFill],fm);
arrow([[615,590],[735,590],[735,553],[791,553]],C.blue);
// Head residual enters a separate, contained mix-and-normalize row.
line([[282,493],[272,493],[272,694],[867,694],[867,665]],C.blue,2.3);arrow([[867,665],[890,665]],C.blue);text('Head residual',475,657,247,28,24,false,C.blue);
box(890,641,151,49,'#FFFFFF',C.green,1.8,8);text('mix α',901,650,128,32,25,false,C.green);arrow([[995,613],[995,641]],C.green);arrow([[1041,665],[1071,665]],C.green);
box(1077,641,127,49,'#FFFFFF',C.green,1.8,8);text('normalize',1083,650,115,32,23,false,C.green);
// AutoAnchor preserves an explicit connection to the fixed semantic prior.
box(1295,318,435,92,'#FFFFFF',C.gold,1.7,10);text('Catalog coverage',1376,323,320,31,27,true,C.gold);bars(1317,334,'#D4A53C');text('sets the prior weight',1374,366,328,28,24,false,C.muted);arrow([[1512,410],[1512,505]],C.gold);text('prior weight',1531,445,178,33,25,false,C.gold);
box(1314,505,402,112,'#FFFFFF',C.gold,2,10);text('Prior interpolation',1333,514,364,35,28,true,C.gold);vec(1350,565,133,22,[C.lilacFill,'#8BBA98','#5B9AD5',C.goldFill],an);text('+',1493,553,39,36,30,true,C.gold);vec(1544,565,133,22,undefined,an);
arrow([[1204,665],[1251,665],[1251,552],[1314,552]],C.green);
// The prior bypass uses the lower margin, clear of module text and outputs.
arrow([[135,584],[135,750],[1290,750],[1290,643],[1578,643],[1578,617]],C.line,2.2);text('Frozen semantic prior',472,754,405,29,25,false,C.muted);
arrow([[1410,617],[1410,659]],C.gold);box(1336,659,148,37,'#FFFFFF',C.gold,1.8,7);text('normalize',1343,664,134,26,23,false,C.gold);arrow([[1484,678],[1516,678]],C.gold);vec(1524,667,167,22,[C.lilacFill,C.coralFill,C.goldFill,C.greenFill,'#5B9AD5'],an);
// Local embeddings and anchored features meet only at the score.
lock(48,797);text('Local histories, profiles and user embeddings',100,793,680,34,25,false,C.muted,'left');user(934,791);text('User embedding',991,797,232,30,25,false,C.blue,'left');arrow([[1223,811],[1311,811]],C.blue);
box(1320,790,397,44,'#F6F9FC',C.line,1.8,8);text('Dot-product ranking  /  BPR',1335,795,368,33,26,true);arrow([[1610,689],[1610,790]],C.gold);
// A compact editable adapter inset and final local optimization.
box(20,878,1010,149,'#EDF5FC',C.blue,2.2,16);text('Inside each adapter',42,888,470,39,30,true,C.ink,'left');text('Two residual blocks, then normalization',42,996,924,27,24,false,C.muted,'left');
vec(52,951,122,21,[C.blueFill,'#5B9AD5',C.lilacFill],{x:28,y:883,w:994,h:137});arrow([[180,962],[233,962]],C.blue);box(242,934,205,57,'#FFFFFF',C.blue,1.8,8);text('Bottleneck MLP',254,947,181,31,25,false,C.blue);shp('ellipse',496,940,45,45,'#FFFFFF',C.blue,1.8);text('+',504,944,29,35,28,false,C.blue);arrow([[447,962],[496,962]],C.blue);arrow([[219,962],[219,931],[518,931],[518,940]],C.blue,1.8);text('× 2',585,939,95,44,30,true,C.blue);arrow([[541,962],[580,962]],C.blue);arrow([[680,962],[725,962]],C.blue);box(735,934,237,57,'#FFFFFF',C.blue,1.8,8);text('Normalize',748,947,211,31,26,false,C.blue);
box(1052,878,728,149,'#FFECEC',C.coral,2.2,16);text('After federation',1077,888,667,39,30,true,C.ink,'left');laptop(1088,950,83,61);text('40 local proximal steps',1192,939,548,35,28,true,C.coral,'left');text('Personalized parameters stay local',1192,981,548,30,25,false,C.muted,'left');
// Source and semantics: no experimental values are encoded in this diagram.
s.speakerNotes.textFrame.setText('Source: paper_www/main.tex, equations for routing, FiLM, and AutoAnchor. Head and tail adapters both train. PGDR uses target training frequency. Private history profiles determine FiLM scale and shift through the shared conditioner. The alpha mix combines conditioned features with the head residual before normalization. Catalog coverage determines prior weight rho_eff; the frozen prior bypasses adaptation. AutoAnchor normalizes its interpolated vector. Each adapter uses two residual blocks followed by normalization. After federation, 40 local proximal steps personalize parameters without upload. All diagram components are native editable shapes; there is no image layer.');
for(const b of bounds){if(b.x<0||b.y<0||b.x+b.w>W+.1||b.y+b.h>H+.1)throw new Error(`Slide overflow ${JSON.stringify(b)}`);}
for(const b of tokens){const o=b.owner;if(b.x<o.x||b.y<o.y||b.x+b.w>o.x+o.w||b.y+b.h>o.y+o.h)throw new Error(`Feature tokens overflow ${JSON.stringify(b)}`);}
const candidate=path.join(build,'candidate-redesigned.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const checkedDir=path.join(root,'.build_www','framework_checked');await fs.mkdir(checkedDir,{recursive:true});const checked=path.join(checkedDir,'redesigned-'+Date.now()+'.pptx');
const report=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:checked,pythonExecutable:'C:/Users/hejie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','17145000,9982200','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation-redesigned-'+Date.now()+'.json')});
await fs.copyFile(checked,final);console.log(JSON.stringify({final,objects:bounds.length,tokenGroups:tokens.length,report}));

// Native PowerPoint export keeps the PNG faithful to the editable deliverable.
if(process.platform==='win32'){
 const psq=v=>"'"+v.replaceAll("'","''")+"'";
 const png=path.join(root,'paper_www','figures','framework_ppt.png');
 const preview=path.join(build,'preview-redesigned.png');
 const script=`$ErrorActionPreference='Stop'
$pptWasRunning=@(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app=New-Object -ComObject PowerPoint.Application
$pres=$null
try {
 $pres=$app.Presentations.Open(${psq(final)},-1,0,0)
 $pres.Slides.Item(1).Export(${psq(png)},'PNG',5400,3144)
 $pres.Slides.Item(1).Export(${psq(preview)},'PNG',1800,1048)
}
finally {
 if($null -ne $pres){$pres.Close();[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pres)}
 if(-not $pptWasRunning -and $app.Presentations.Count -eq 0){$app.Quit()}
 [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
}`;
 const result=spawnSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',script],{encoding:'utf8',windowsHide:true,timeout:120000});
 if(result.error||result.status!==0)throw new Error(`PowerPoint export failed: ${result.error||result.stderr}`);
 console.log(JSON.stringify({png,preview,width:5400,height:3144}));
}

