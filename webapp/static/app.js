(() => {
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const runBtn = document.getElementById('runBtn');
  const clearBtn = document.getElementById('clearBtn');
  const modeSel = document.getElementById('mode');
  const multimaskChk = document.getElementById('multimask');
  let withDepth = true;
  const refineDepthChk = document.getElementById('refineDepth');
  const growRegionChk = document.getElementById('growRegion');
  const depthTol = document.getElementById('depthTol');
  const depthTolVal = document.getElementById('depthTolVal');
  const exportPlyChk = document.getElementById('exportPly');
  const withSemanticChk = document.getElementById('withSemantic');
  const wallDemoChk = document.getElementById('wallDemo');
  depthTol.addEventListener('input', () => depthTolVal.textContent = Number(depthTol.value).toFixed(2));
  const resultsEl = document.getElementById('results');
  const progressEl = document.getElementById('progress');
  const palettePanel = document.getElementById('palettePanel');
  const paletteGrid = document.getElementById('paletteGrid');
  const customColorInput = document.getElementById('customColor');
  const maskInfo = document.getElementById('maskInfo');
  const maskName = document.getElementById('maskName');
  const maskCount = document.getElementById('maskCount');
  const colorizeBtn = document.getElementById('colorizeBtn');
  const undoBtn = document.getElementById('undoBtn');
  const redoBtn = document.getElementById('redoBtn');
  
  let selectedMaskUrls = [];
  let selectedColor = '#4f8cff';
  let alphaValue = 0.6;
  const alphaSlider = document.getElementById('alphaSlider');
  const alphaVal = document.getElementById('alphaVal');
  
  // History: store image URLs for undo/redo
  let colorizeHistory = [];  // [{type: 'layer1'|'mask', url: original, colorizedUrl: result}]
  let historyIndex = -1;
  let currentLayer1Url = null;  // Track current Layer1 URL
  
  alphaSlider.addEventListener('input', () => {
    alphaValue = Number(alphaSlider.value) / 100;
    alphaVal.textContent = `${alphaSlider.value}%`;
  });
  
  // Initialize palette
  const paletteColors = [
    '#4f8cff', '#ff4f4f', '#4fff4f', '#ffff4f', '#ff4fff', '#4fffff',
    '#ff8c4f', '#8c4fff', '#4fff8c', '#ff4f8c', '#8cff4f', '#4f8cff',
    '#ffffff', '#000000', '#808080', '#ff8080', '#80ff80', '#8080ff',
  ];
  paletteColors.forEach((color) => {
    const btn = document.createElement('button');
    btn.className = 'palette-color';
    btn.style.backgroundColor = color;
    btn.dataset.color = color;
    btn.title = color;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.palette-color').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      selectedColor = color;
      customColorInput.value = color;
      // Enable button if mask/Layer1 is already selected
      if (selectedMaskUrls.length > 0) {
        colorizeBtn.disabled = false;
      }
    });
    paletteGrid.appendChild(btn);
  });
  customColorInput.addEventListener('change', (e) => {
    selectedColor = e.target.value;
    document.querySelectorAll('.palette-color').forEach(b => b.classList.remove('selected'));
    // Enable button if mask/Layer1 is already selected
    if (selectedMaskUrls.length > 0) {
      colorizeBtn.disabled = false;
    }
  });

  let img = new Image();
  let imgNaturalW = 0, imgNaturalH = 0;
  let scaleX = 1, scaleY = 1;
  let lastDataUrl = null;
  let points = [];
  let box = null;
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
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x/scaleX, p.y/scaleY, 6, 0, Math.PI*2);
      ctx.fillStyle = p.label === 1 ? 'rgba(0,255,0,0.9)' : 'rgba(255,0,0,0.9)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#000';
      ctx.stroke();
    }
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
      lastDataUrl = e.target.result;
      img.onload = () => {
        imgNaturalW = img.naturalWidth;
        imgNaturalH = img.naturalHeight;
        setCanvasSizeToImage();
        draw();
      };
      img.src = lastDataUrl;
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

  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  canvas.addEventListener('mousedown', (e) => {
    if (modeSel.value === 'box') {
      if (!e.shiftKey) return;
      isDraggingBox = true;
      dragStart = {x: e.offsetX * scaleX, y: e.offsetY * scaleY};
      box = {x1: dragStart.x, y1: dragStart.y, x2: dragStart.x, y2: dragStart.y};
      draw();
      return;
    }
    if (modeSel.value !== 'points') return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const label = e.button === 0 ? 1 : 0;
    points.push({x, y, label});
    draw();
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDraggingBox || !box) return;
    box.x2 = e.offsetX * scaleX;
    box.y2 = e.offsetY * scaleY;
    draw();
  });
  canvas.addEventListener('mouseup', () => {
    isDraggingBox = false;
  });

  clearBtn.addEventListener('click', () => {
    points = [];
    box = null;
    draw();
  });

  async function run() {
    if (!lastDataUrl) {
      alert('이미지를 먼저 업로드하세요.');
      return;
    }
    const fd = new FormData();
    let blob;
    try {
      const blobRes = await fetch(lastDataUrl);
      if (!blobRes.ok) throw new Error('Failed to load image');
      blob = await blobRes.blob();
    } catch (e) {
      alert('이미지 로드 오류: ' + e.message);
      return;
    }
    fd.append('file', blob, 'image.jpg');
    fd.append('mode', modeSel.value);
    if (modeSel.value === 'points' && points.length) {
      fd.append('points', JSON.stringify(points));
    }
    if (modeSel.value === 'box' && box) {
      fd.append('box', JSON.stringify(box));
    }
    fd.append('with_depth', withDepth ? 'true' : 'false');
    fd.append('refine_depth', refineDepthChk.checked ? 'true' : 'false');
    fd.append('grow_region', growRegionChk.checked ? 'true' : 'false');
    fd.append('depth_tolerance', String(depthTol.value));
    fd.append('export_ply', exportPlyChk.checked ? 'true' : 'false');
    fd.append('with_semantic', withSemanticChk.checked ? 'true' : 'false');
    fd.append('wall_demo', wallDemoChk.checked ? 'true' : 'false');
    fd.append('wall_depth_delta', String(depthTol.value));
    progressEl.hidden = false;
    resultsEl.hidden = true;
    resultsEl.innerHTML = '';
    colorizeHistory = [];
    historyIndex = -1;
    undoBtn.disabled = true;
    redoBtn.disabled = true;
    // Reset selection state
    selectedMaskUrls = [];
    maskName.textContent = '-';
    maskCount.textContent = '0';
    maskInfo.hidden = false;  // Keep visible but show "no selection"
    colorizeBtn.disabled = true;
    try {
      const res = await fetch('/api/segment', { method: 'POST', body: fd });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errorText}`);
      }
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'failed');
      const overlays = data.overlays || [];
      const masks = data.masks || [];
      const depthUrl = data.depth;
      const plyUrl = data.ply;
      const semanticUrl = data.semantic;
      const semanticOverlayUrl = data.semantic_overlay;
      const semanticLabels = data.semantic_labels || [];
      const demo = data.demo;
      const frag = document.createDocumentFragment();
      const grid = document.createElement('div');
      grid.className = 'grid';

      let tilesAdded = 0;

      if (lastDataUrl && tilesAdded < 6) {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = 'Original';
        const imgEl = document.createElement('img');
        imgEl.src = lastDataUrl;
        card.appendChild(h);
        card.appendChild(imgEl);
        grid.appendChild(card);
        tilesAdded++;
      }

      if (depthUrl && tilesAdded < 6) {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = 'Depth Map';
        const dimg = document.createElement('img');
        dimg.src = depthUrl;
        card.appendChild(h);
        card.appendChild(dimg);
        grid.appendChild(card);
        tilesAdded++;
      }

      if (semanticOverlayUrl && tilesAdded < 6) {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = 'Semantic (ADE20K)';
        const simg = document.createElement('img');
        simg.src = semanticOverlayUrl;
        const list = document.createElement('div');
        list.className = 'links';
        list.style.fontSize = '11px';
        if (semanticLabels.length) {
          const lines = semanticLabels.slice(0, 6).map(x => `${x.label}(${Math.round(x.ratio*100)}%)`);
          list.textContent = lines.join(' / ');
        } else {
          list.textContent = '(labels unavailable)';
        }
        const dl = document.createElement('div');
        dl.className = 'links';
        dl.style.fontSize = '10px';
        if (semanticUrl) {
          dl.innerHTML = `<a href="${semanticOverlayUrl}" download>overlay</a> | <a href="${semanticUrl}" download>map</a>`;
        } else {
          dl.innerHTML = `<a href="${semanticOverlayUrl}" download>overlay</a>`;
        }
        card.appendChild(h);
        card.appendChild(simg);
        card.appendChild(list);
        card.appendChild(dl);
        grid.appendChild(card);
        tilesAdded++;
      }

      if (demo && tilesAdded < 6) {
        const makeImgCard = (title, url, isLayer1 = false) => {
          const card = document.createElement('div');
          card.className = 'card';
          if (isLayer1) {
            currentLayer1Url = url;
            card.dataset.layer1Url = url;
            card.style.cursor = 'pointer';
            card.style.border = '1px solid #1b2740';
            card.title = '클릭하여 색칠하기';
            card.addEventListener('click', (e) => {
              document.querySelectorAll('.card').forEach(c => {
                if (c !== card) {
                  c.style.border = '1px solid #1b2740';
                  c.classList.remove('selected');
                }
              });
              card.style.border = '2px solid #4f8cff';
              card.classList.add('selected');
              selectedMaskUrls = [url];
              selectedMaskUrls = [url];
              maskName.textContent = 'Layer1';
              maskCount.textContent = '1';
              maskInfo.hidden = false;
              // Enable colorize button when Layer1 is selected (color can be selected later)
              colorizeBtn.disabled = false;
            });
          }
          const h = document.createElement('h3');
          h.textContent = title;
          const im = document.createElement('img');
          im.src = url;
          const dl = document.createElement('div');
          dl.className = 'links';
          dl.style.fontSize = '10px';
          dl.innerHTML = `<a href="${url}" download>${title.toLowerCase()}.png</a>`;
          card.appendChild(h);
          card.appendChild(im);
          card.appendChild(dl);
          return card;
        };
        if (demo.layer1 && tilesAdded < 6) { 
          grid.appendChild(makeImgCard('Layer1', demo.layer1, true)); 
          tilesAdded++; 
        }
        if (demo.layer2 && tilesAdded < 6) { 
          grid.appendChild(makeImgCard('Layer2', demo.layer2)); 
          tilesAdded++; 
        }
        if (demo.overlay && tilesAdded < 6) { 
          grid.appendChild(makeImgCard('Overlay', demo.overlay)); 
          tilesAdded++; 
        }
      }

      overlays.forEach((ov, idx) => {
        if (tilesAdded >= 6) return;
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.maskUrl = masks[idx];
        card.style.cursor = 'pointer';
        card.style.border = '1px solid #1b2740';
        card.title = '클릭하여 색칠하기 (Ctrl: 다중 선택)';
        card.addEventListener('click', (e) => {
          document.querySelectorAll('.card[data-layer1-url]').forEach(c => {
            c.style.border = '1px solid #1b2740';
            c.classList.remove('selected');
          });
          if (e.ctrlKey || e.metaKey) {
            if (selectedMaskUrls.includes(masks[idx])) {
              selectedMaskUrls = selectedMaskUrls.filter(u => u !== masks[idx]);
              card.style.border = '1px solid #1b2740';
              card.classList.remove('selected');
            } else {
              selectedMaskUrls.push(masks[idx]);
              card.style.border = '2px solid #4f8cff';
              card.classList.add('selected');
            }
          } else {
            document.querySelectorAll('.card').forEach(c => {
              if (!c.dataset.layer1Url) {
                c.style.border = '1px solid #1b2740';
                c.classList.remove('selected');
              }
            });
            selectedMaskUrls = [masks[idx]];
            card.style.border = '2px solid #4f8cff';
            card.classList.add('selected');
          }
          maskCount.textContent = selectedMaskUrls.length;
          if (selectedMaskUrls.length > 0) {
            maskName.textContent = selectedMaskUrls.length === 1 
              ? `mask_${idx}.png` 
              : `${selectedMaskUrls.length}개 선택됨`;
            maskInfo.hidden = false;
            // Enable button when mask is selected (color can be selected later)
            colorizeBtn.disabled = false;
          } else {
            maskInfo.hidden = true;
            colorizeBtn.disabled = true;
          }
        });
        const h = document.createElement('h3');
        h.textContent = `Result ${idx+1}`;
        const imgOv = document.createElement('img');
        imgOv.src = ov;
        const imgMask = document.createElement('img');
        imgMask.src = masks[idx];
        imgMask.style.maxHeight = '80px';
        const links = document.createElement('div');
        links.className = 'links';
        links.style.fontSize = '10px';
        links.innerHTML = `<a href="${ov}" download>overlay</a> | <a href="${masks[idx]}" download>mask</a>`;
        card.appendChild(h);
        card.appendChild(imgOv);
        card.appendChild(imgMask);
        card.appendChild(links);
        grid.appendChild(card);
        tilesAdded++;
      });

      if (plyUrl) {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = 'Point Cloud (PLY)';
        const link = document.createElement('a');
        link.href = plyUrl;
        link.download = 'cloud.ply';
        link.textContent = 'cloud.ply 다운로드';
        card.appendChild(h);
        card.appendChild(link);
        grid.appendChild(card);
      }
      frag.appendChild(grid);
      resultsEl.appendChild(frag);
      resultsEl.hidden = false;
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      alert('오류: ' + errorMsg);
      console.error('Segment error:', e);
    } finally {
      progressEl.hidden = true;
    }
  }
  runBtn.addEventListener('click', run);
  
  // Colorize functionality
  colorizeBtn.addEventListener('click', async () => {
    if (!selectedMaskUrls.length) {
      alert('마스크 또는 Layer1을 먼저 선택해주세요.');
      return;
    }
    if (!selectedColor) {
      alert('색상을 선택해주세요.');
      return;
    }
    progressEl.hidden = false;
    colorizeBtn.disabled = true;
    try {
      const hex = selectedColor.replace('#', '');
      const r = parseInt(hex.substr(0, 2), 16);
      const g = parseInt(hex.substr(2, 2), 16);
      const b = parseInt(hex.substr(4, 2), 16);
      const fd = new FormData();
      
      const isLayer1Selected = Array.from(resultsEl.querySelectorAll('.card')).some(c => {
        return c.dataset.layer1Url && selectedMaskUrls.includes(c.dataset.layer1Url);
      });
      
      // Save to history BEFORE applying
      const originalUrl = isLayer1Selected ? currentLayer1Url : selectedMaskUrls[0];
      const historyEntry = {
        type: isLayer1Selected ? 'layer1' : 'mask',
        originalUrl: originalUrl,
        colorizedUrl: null,
        masks: [...selectedMaskUrls],
        color: selectedColor,
        alpha: alphaValue
      };
      
      if (isLayer1Selected) {
        fd.append('layer1_url', selectedMaskUrls[0]);
      } else {
        fd.append('mask_urls', JSON.stringify(selectedMaskUrls));
        const originalCard = Array.from(resultsEl.querySelectorAll('.card')).find(c => {
          const h3 = c.querySelector('h3');
          return h3 && h3.textContent === 'Original';
        });
        if (originalCard) {
          const origImg = originalCard.querySelector('img');
          if (origImg && origImg.src.startsWith('/static/')) {
            fd.append('base_image_url', origImg.src);
          }
        }
      }
      
      fd.append('color_r', String(r));
      fd.append('color_g', String(g));
      fd.append('color_b', String(b));
      fd.append('alpha', String(alphaValue));
      
      const res = await fetch('/api/colorize', { method: 'POST', body: fd });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errorText}`);
      }
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'failed');
      
      historyEntry.colorizedUrl = data.result_url;
      colorizeHistory = colorizeHistory.slice(0, historyIndex + 1);
      colorizeHistory.push(historyEntry);
      historyIndex = colorizeHistory.length - 1;
      undoBtn.disabled = false;
      redoBtn.disabled = true;
      
      if (data.type === 'layer1') {
        const layer1Card = Array.from(resultsEl.querySelectorAll('.card')).find(c => {
          return c.dataset.layer1Url && c.dataset.layer1Url === selectedMaskUrls[0];
        });
        if (layer1Card) {
          const img = layer1Card.querySelector('img');
          if (img) {
            currentLayer1Url = data.result_url;
            layer1Card.dataset.layer1Url = data.result_url;  // Update URL
            img.src = data.result_url + '?t=' + Date.now();
            const dl = layer1Card.querySelector('.links');
            if (dl) {
              dl.innerHTML = `<a href="${data.result_url}" download>layer1_colorized_${data.timestamp}.png</a>`;
            }
          }
        }
      } else {
        let existingCard = resultsEl.querySelector('.card[data-colorized="true"]');
        if (!existingCard) {
          existingCard = document.createElement('div');
          existingCard.className = 'card';
          existingCard.dataset.colorized = 'true';
          const h = document.createElement('h3');
          h.textContent = 'Colorized';
          existingCard.appendChild(h);
          const img = document.createElement('img');
          existingCard.appendChild(img);
          const dl = document.createElement('div');
          dl.className = 'links';
          dl.style.fontSize = '10px';
          existingCard.appendChild(dl);
          const grid = resultsEl.querySelector('.grid');
          if (grid && grid.children.length < 6) {
            grid.appendChild(existingCard);
          }
        }
        const img = existingCard.querySelector('img');
        const dl = existingCard.querySelector('.links');
        img.src = data.result_url + '?t=' + Date.now();
        dl.innerHTML = `<a href="${data.result_url}" download>colorized_${data.timestamp}.png</a>`;
      }
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      alert('색칠 오류: ' + errorMsg);
      console.error('Colorize error:', e);
    } finally {
      progressEl.hidden = true;
      colorizeBtn.disabled = false;
    }
  });
  
  // Undo/Redo with actual image restoration
  undoBtn.addEventListener('click', () => {
    if (historyIndex > 0) {
      historyIndex--;
      const state = colorizeHistory[historyIndex];
      
      // Restore image
      if (state.type === 'layer1') {
        const layer1Card = Array.from(resultsEl.querySelectorAll('.card')).find(c => {
          return c.dataset.layer1Url;
        });
        if (layer1Card) {
          const img = layer1Card.querySelector('img');
          if (img) {
            currentLayer1Url = state.originalUrl;
            layer1Card.dataset.layer1Url = state.originalUrl;
            img.src = state.originalUrl + '?t=' + Date.now();
          }
        }
      } else {
        // For masks, would need to restore original mask result
        // For now, just restore selection state
      }
      
      // Restore UI state
      selectedMaskUrls = [...state.masks];
      selectedColor = state.color;
      alphaValue = state.alpha;
      alphaSlider.value = Math.round(alphaValue * 100);
      alphaVal.textContent = `${alphaSlider.value}%`;
      customColorInput.value = selectedColor;
      document.querySelectorAll('.palette-color').forEach(b => {
        b.classList.toggle('selected', b.dataset.color === selectedColor);
      });
      
      undoBtn.disabled = historyIndex <= 0;
      redoBtn.disabled = false;
    }
  });
  
  redoBtn.addEventListener('click', () => {
    if (historyIndex < colorizeHistory.length - 1) {
      historyIndex++;
      const state = colorizeHistory[historyIndex];
      
      // Restore image
      if (state.type === 'layer1' && state.colorizedUrl) {
        const layer1Card = Array.from(resultsEl.querySelectorAll('.card')).find(c => {
          return c.dataset.layer1Url;
        });
        if (layer1Card) {
          const img = layer1Card.querySelector('img');
          if (img) {
            currentLayer1Url = state.colorizedUrl;
            layer1Card.dataset.layer1Url = state.colorizedUrl;
            img.src = state.colorizedUrl + '?t=' + Date.now();
          }
        }
      }
      
      // Restore UI state
      selectedMaskUrls = [...state.masks];
      selectedColor = state.color;
      alphaValue = state.alpha;
      alphaSlider.value = Math.round(alphaValue * 100);
      alphaVal.textContent = `${alphaSlider.value}%`;
      customColorInput.value = selectedColor;
      document.querySelectorAll('.palette-color').forEach(b => {
        b.classList.toggle('selected', b.dataset.color === selectedColor);
      });
      
      undoBtn.disabled = false;
      redoBtn.disabled = historyIndex >= colorizeHistory.length - 1;
    }
  });
})();
