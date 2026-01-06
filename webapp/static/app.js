(() => {
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const runBtn = document.getElementById('runBtn');
  const clearBtn = document.getElementById('clearBtn');
  const modeSel = document.getElementById('mode');
  const multimaskChk = document.getElementById('multimask');
  const resultsEl = document.getElementById('results');
  const progressEl = document.getElementById('progress');

  let img = new Image();
  let imgNaturalW = 0, imgNaturalH = 0;
  let scaleX = 1, scaleY = 1;
  let points = []; // {x,y,label}
  let box = null;  // {x1,y1,x2,y2}
  let isDraggingBox = false;
  let dragStart = null;

  function setCanvasSizeToImage() {
    const maxW = 960;
    const ratio = Math.min(1, maxW / imgNaturalW);
    const w = Math.round(imgNaturalW * ratio);
    const h = Math.round(imgNaturalH * ratio);
    canvas.width = w; canvas.height = h;
    scaleX = imgNaturalW / w;
    scaleY = imgNaturalH / h;
  }

  function draw() {
    if (!imgNaturalW) return;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    // draw points
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x/scaleX, p.y/scaleY, 6, 0, Math.PI*2);
      ctx.fillStyle = p.label === 1 ? 'rgba(0,255,0,0.9)' : 'rgba(255,0,0,0.9)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#000';
      ctx.stroke();
    }
    // draw box
    if (box) {
      const x = Math.min(box.x1, box.x2)/scaleX;
      const y = Math.min(box.y1, box.y2)/scaleY;
      const w = Math.abs(box.x2 - box.x1)/scaleX;
      const h = Math.abs(box.y2 - box.y1)/scaleY;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#4f8cff';
      ctx.strokeRect(x, y, w, h);
    }
  }

  function setFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      img.onload = () => {
        imgNaturalW = img.naturalWidth;
        imgNaturalH = img.naturalHeight;
        setCanvasSizeToImage();
        draw();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    points = []; box = null;
    const f = e.target.files[0];
    setFile(f);
  });

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault(); dropzone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    points = []; box = null;
    setFile(f);
  });

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    if (!imgNaturalW) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    if (modeSel.value === 'box' && e.shiftKey) {
      isDraggingBox = true;
      dragStart = {x, y};
      box = {x1:x, y1:y, x2:x, y2:y};
      draw();
    }
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDraggingBox || !dragStart) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    box.x2 = x; box.y2 = y;
    draw();
  });
  canvas.addEventListener('mouseup', (e) => {
    if (isDraggingBox) {
      isDraggingBox = false;
      dragStart = null;
      draw();
      return;
    }
    if (modeSel.value !== 'points') return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const label = (e.button === 2) ? 0 : 1; // right=neg
    points.push({x, y, label});
    draw();
  });

  clearBtn.addEventListener('click', () => {
    points = []; box = null;
    draw();
  });

  async function run() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) { alert('이미지를 선택하세요.'); return; }
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', modeSel.value);
    fd.append('multimask', multimaskChk.checked ? 'true' : 'false');
    if (modeSel.value === 'points' && points.length) {
      fd.append('points', JSON.stringify(points));
    }
    if (modeSel.value === 'box' && box) {
      fd.append('box', JSON.stringify(box));
    }
    progressEl.hidden = false;
    resultsEl.hidden = true;
    resultsEl.innerHTML = '';
    try {
      const res = await fetch('/api/segment', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'failed');
      const overlays = data.overlays || [];
      const masks = data.masks || [];
      const frag = document.createDocumentFragment();
      const grid = document.createElement('div');
      grid.className = 'grid';
      overlays.forEach((ov, idx) => {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = `Result ${idx+1}`;
        const imgOv = document.createElement('img');
        imgOv.src = ov;
        const imgMask = document.createElement('img');
        imgMask.src = masks[idx];
        const links = document.createElement('div');
        links.className = 'links';
        links.innerHTML = `<a href=\"${ov}\" download>overlay.png</a> | <a href=\"${masks[idx]}\" download>mask.png</a>`;
        card.appendChild(h);
        card.appendChild(imgOv);
        card.appendChild(imgMask);
        card.appendChild(links);
        grid.appendChild(card);
      });
      frag.appendChild(grid);
      resultsEl.appendChild(frag);
      resultsEl.hidden = false;
    } catch (e) {
      alert('오류: ' + e.message);
    } finally {
      progressEl.hidden = true;
    }
  }
  runBtn.addEventListener('click', run);
})(); 

