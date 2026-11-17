
(() => {
  const root = document.getElementById('uvg-unit-construction');
  const payload = JSON.parse(root.querySelector('#uvg-unit-data').textContent);
  const sceneSelect = root.querySelector('#uvg-scene');
  const stageSelect = root.querySelector('#uvg-stage');
  const unitSelect = root.querySelector('#uvg-unit');
  const original = root.querySelector('#uvg-original');
  const current = root.querySelector('#uvg-current');
  const title = root.querySelector('#uvg-current-label');
  const detail = root.querySelector('#uvg-selected');
  const legend = root.querySelector('#uvg-legend');
  const vectorTitle = root.querySelector('#uvg-vector-title');
  const vectorSvg = root.querySelector('#uvg-vector');
  const stages = [
    '28×28 图块，每块 16×16 像素',
    'DINOv3 特征：前三个主成分投影',
    '第1轮注意力分配', '第2轮注意力分配', '第3轮注意力分配',
    '最终解码器区域权重', '选定单元的软区域权重'
  ];
  let state = { scene: 0, stage: 0, unit: payload.scenes[0].defaultUnit };
  const decode = text => Uint8Array.from(atob(text), ch => ch.charCodeAt(0));
  const scenes = payload.scenes.map(s => {
    const image = new Image(), feature = new Image();
    const out = {...s, imageObject: image, featureObject: feature, weights: decode(s.weights), hard: decode(s.hard)};
    image.onload = () => draw();
    feature.onload = () => draw();
    image.src = s.image; feature.src = s.featureImage;
    return out;
  });
  scenes.forEach((scene,index) => sceneSelect.add(new Option(scene.city,index)));
  for (let i=0;i<32;i++) unitSelect.add(new Option(`U${String(i+1).padStart(2,'0')}`,i));
  function restore(snapshot) {
    const saved = snapshot && snapshot.privateContent;
    if (!saved) return;
    for (const key of ['scene','stage','unit']) {
      const limit = key === 'scene' ? scenes.length : key === 'stage' ? 7 : 32;
      if (Number.isInteger(saved[key]) && saved[key]>=0 && saved[key]<limit) state[key]=saved[key];
    }
  }
  restore(window.openai && window.openai.widgetState);
  function persist() {
    if (window.openai && window.openai.setWidgetState) {
      window.openai.setWidgetState({
        modelContent: {scene: scenes[state.scene].city, stage: stages[state.stage], unit: `U${state.unit+1}`, source: 'trained DINOv3 pilot inference'},
        privateContent: {...state}
      }).catch(() => {});
    }
  }
  function rgb(variable) {
    const probe=document.createElement('span');
    probe.style.color=`var(${variable})`;root.appendChild(probe);
    const value=getComputedStyle(probe).color;probe.remove();return value;
  }
  function weightIndex() { return state.stage >= 5 ? 3 : Math.max(0,state.stage-2); }
  function vectorIndex() { return state.stage < 2 ? 0 : Math.min(3,state.stage-1); }
  function drawVector(scene) {
    const values=scene.states[vectorIndex()][state.unit];
    const description=state.stage<2 ? '初始化候选向量' : state.stage<5 ? `第${state.stage-1}轮更新后的向量` : '最终单元候选向量';
    vectorTitle.textContent=`${description} · 96维`;
    if (typeof d3 === 'undefined') { vectorTitle.textContent += ` · 前6维 ${values.slice(0,6).join('，')}`; return; }
    const width=vectorSvg.getBoundingClientRect().width || root.clientWidth;
    const height=142, left=58, right=12, top=12, bottom=36;
    const svg=d3.select(vectorSvg);svg.selectAll('*').remove();svg.attr('viewBox',`0 0 ${width} ${height}`).attr('height',height);
    svg.append('title').text(`${description}，U${state.unit+1}，96个实测坐标值`);
    const extent=d3.extent(scene.states.flat(2));
    const min=Math.min(0,extent[0]),max=Math.max(0,extent[1]),padding=(max-min)*.05||1;
    const x=d3.scaleLinear().domain([0,96]).range([left,width-right]);
    const y=d3.scaleLinear().domain([min-padding,max+padding]).range([height-bottom,top]);
    svg.append('rect').attr('data-chart-frame','').attr('x',left).attr('y',top).attr('width',width-left-right).attr('height',height-top-bottom).attr('fill','none').attr('stroke','var(--border)');
    svg.append('line').attr('x1',left).attr('x2',width-right).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','var(--border)');
    values.forEach((value,index) => svg.append('rect').attr('x',x(index)+.3).attr('width',Math.max(.5,x(1)-x(0)-.6)).attr('y',Math.min(y(0),y(value))).attr('height',Math.max(.5,Math.abs(y(value)-y(0)))).attr('fill','var(--viz-series-1)').attr('opacity',value>=0?1:.55));
    const label=(xpos,ypos,text,anchor='middle') => svg.append('text').attr('x',xpos).attr('y',ypos).attr('text-anchor',anchor).attr('fill','var(--foreground)').attr('font-size',12).text(text);
    const ticks=width<400?[1,32,64,96]:[1,24,48,72,96];
    ticks.forEach(n => label(x(n-.5),height-bottom+17,n,n===1?'start':n===96?'end':'middle'));
    [min,0,max].forEach(value => label(left-8,y(value)+4,value.toFixed(1),'end'));
    label((left+width-right)/2,height-3,'维度编号').attr('class','axis-title').attr('data-axis','x');
    label(left-8,10,'数值','end').attr('class','axis-title').attr('data-axis','y');
    svg.append('rect').attr('data-chart-hit','').attr('x',left).attr('y',top).attr('width',width-left-right).attr('height',height-top-bottom).attr('fill','transparent')
      .on('pointermove',event => {
        const point=d3.pointer(event,vectorSvg);
        const index=Math.max(0,Math.min(95,Math.floor(x.invert(point[0]))));
        vectorTitle.textContent=`${description} · 第${index+1}维：${values[index].toFixed(3)}`;
      }).on('pointerleave',() => {vectorTitle.textContent=`${description} · 96维`;});
  }
  function draw() {
    const scene=scenes[state.scene];
    sceneSelect.value=String(state.scene);stageSelect.value=String(state.stage);unitSelect.value=String(state.unit);
    title.textContent=stages[state.stage];
    const ctx=current.getContext('2d'), raw=original.getContext('2d');
    ctx.clearRect(0,0,448,448);raw.clearRect(0,0,448,448);
    if (scene.imageObject.complete && scene.imageObject.naturalWidth) {
      raw.drawImage(scene.imageObject,0,0,448,448);ctx.drawImage(scene.imageObject,0,0,448,448);
    }
    const shade=rgb('--background'),stroke=rgb('--foreground'),color=rgb('--viz-series-1');
    const map=weightIndex(),offset=map*784;
    const ids=scene.hard.subarray(offset,offset+784);
    const counts=new Array(32).fill(0);ids.forEach(i => counts[i]++);
    const max=scene.maxWeights[map][state.unit];
    if (state.stage===0) {
      ctx.strokeStyle=stroke;ctx.globalAlpha=.4;ctx.lineWidth=.5;ctx.beginPath();
      for(let n=1;n<28;n++){ctx.moveTo(n*16,0);ctx.lineTo(n*16,448);ctx.moveTo(0,n*16);ctx.lineTo(448,n*16);}ctx.stroke();ctx.globalAlpha=1;
      legend.textContent='空间网格：784 个图块';
    } else if(state.stage===1) {
      if(scene.featureObject.complete && scene.featureObject.naturalWidth) ctx.drawImage(scene.featureObject,0,0,448,448);
      legend.textContent='RGB 分别对应 PC1、PC2、PC3；颜色不代表对象类别';
    } else if(state.stage===6) {
      const base=(map*32+state.unit)*784;
      for(let p=0;p<784;p++) {
        const relative=max>0?Math.min(1,scene.weights[base+p]/255/max):0;
        ctx.fillStyle=shade;ctx.globalAlpha=.94*(1-relative);ctx.fillRect((p%28)*16,Math.floor(p/28)*16,16,16);
      }
      ctx.globalAlpha=1;
      legend.textContent=`亮度对应选定单元的软权重：0 → ${max.toFixed(3)}`;
    } else {
      for(let p=0;p<784;p++) {
        ctx.fillStyle=ids[p]===state.unit?color:shade;
        ctx.globalAlpha=ids[p]===state.unit?.38:.18;ctx.fillRect((p%28)*16,Math.floor(p/28)*16,16,16);
      }
      ctx.globalAlpha=.7;ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.beginPath();
      for(let y=0;y<28;y++) for(let x=0;x<28;x++) {
        const p=y*28+x;
        if(x<27 && ids[p]!==ids[p+1]) {ctx.moveTo((x+1)*16,y*16);ctx.lineTo((x+1)*16,(y+1)*16);}
        if(y<27 && ids[p]!==ids[p+28]) {ctx.moveTo(x*16,(y+1)*16);ctx.lineTo((x+1)*16,(y+1)*16);}
      }ctx.stroke();ctx.globalAlpha=1;
      legend.textContent='线条：每个图块的最大权重分组；高亮：选定候选单元';
    }
    detail.textContent=state.stage<2 ? `U${String(state.unit+1).padStart(2,'0')} · 32 个初始化候选之一 · 96维` : `U${String(state.unit+1).padStart(2,'0')} · 硬分配 ${counts[state.unit]}/784 个图块 · 最大软权重 ${max.toFixed(3)}`;
    current.setAttribute('aria-label',`${scene.city}，${stages[state.stage]}，选定候选U${state.unit+1}`);
    drawVector(scene);
  }
  sceneSelect.addEventListener('change',() => {state.scene=Number(sceneSelect.value);state.unit=scenes[state.scene].defaultUnit;draw();persist();});
  stageSelect.addEventListener('change',() => {state.stage=Number(stageSelect.value);draw();persist();});
  unitSelect.addEventListener('change',() => {state.unit=Number(unitSelect.value);draw();persist();});
  current.addEventListener('click',event => {
    if(state.stage<2) return;
    const box=current.getBoundingClientRect();
    const x=Math.max(0,Math.min(27,Math.floor((event.clientX-box.left)/box.width*28)));
    const y=Math.max(0,Math.min(27,Math.floor((event.clientY-box.top)/box.height*28)));
    state.unit=scenes[state.scene].hard[weightIndex()*784+y*28+x];draw();persist();
  });
  window.addEventListener('openai:set_globals',event => {restore(event.detail && event.detail.globals && event.detail.globals.widgetState);draw();});
  new ResizeObserver(() => draw()).observe(root);
  new MutationObserver(() => draw()).observe(document.documentElement,{attributes:true,attributeFilter:['class','style','data-theme']});
  draw();
})();
