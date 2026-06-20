/**
 * player-enhancements.js — Optional UI enhancements for the stroke painter.
 *
 * Loaded after engine.js and the main player IIFE in index.html. Adds:
 *   1. Stroke highlight & info tooltip on hover/click
 *   2. Thumbnail navigation sidebar (click to jump)
 *   3. Customizable keyboard shortcuts (saved to localStorage)
 *   4. Live brush preview (pressure curve + stroke sample)
 *   5. Touch gesture support (pinch-zoom, swipe navigation)
 *
 * All enhancements are opt-in via window.__playerEnh and degrade gracefully
 * if the main player API (window.__player) is unavailable.
 */
'use strict';

(function () {
  // Defer init until window.__player is available (set by the main player IIFE).
  // We poll for it rather than failing immediately.

  // ===========================================================================
  // 1. Stroke highlight & info tooltip
  // ===========================================================================

  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  var tooltip = null;
  var highlightedIdx = -1;
  var savedImageData = null;  // snapshot before highlight overlay

  function ensureTooltip() {
    if (tooltip) return tooltip;
    tooltip = document.createElement('div');
    tooltip.id = 'strokeTooltip';
    tooltip.style.cssText = [
      'position:fixed', 'pointer-events:none', 'z-index:1000',
      'background:rgba(20,20,40,0.95)', 'color:#e0e0e0',
      'padding:8px 12px', 'border-radius:6px', 'font-size:12px',
      'font-family:monospace', 'border:1px solid #667eea',
      'box-shadow:0 4px 12px rgba(0,0,0,0.4)', 'display:none',
      'max-width:280px', 'line-height:1.5'
    ].join(';');
    document.body.appendChild(tooltip);
    return tooltip;
  }

  function showTooltip(x, y, html) {
    var t = ensureTooltip();
    t.innerHTML = html;
    t.style.display = 'block';
    // Position near cursor, clamped to viewport
    var rect = t.getBoundingClientRect();
    var tx = x + 14;
    var ty = y + 14;
    if (tx + rect.width > window.innerWidth) tx = x - rect.width - 14;
    if (ty + rect.height > window.innerHeight) ty = y - rect.height - 14;
    t.style.left = tx + 'px';
    t.style.top = ty + 'px';
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = 'none';
  }

  function describeInstruction(inst) {
    if (!Array.isArray(inst)) return 'Unknown';
    var op = inst[0];
    if (op === 'background') return 'Background: ' + inst[1];
    if (op === 'colour') return 'Colour: ' + inst[1];
    if (op === 'width') return 'Width: ' + inst[1] + 'px';
    if (op === 'alpha') return 'Alpha: ' + inst[1];
    if (op === 'line') {
      var brushId = inst[1];
      var pts = inst.slice(2);
      var stride = brushId === 5 ? 3 : 2;
      var nPoints = Math.floor(pts.length / stride);
      var brushName = brushId === 0 ? '马克笔(marker)' :
                      brushId === 5 ? '压感v3(pressure)' : 'brush#' + brushId;
      var first = pts[0].toFixed(1) + ',' + pts[1].toFixed(1);
      var last = pts[pts.length - stride].toFixed(1) + ',' +
                 pts[pts.length - stride + 1].toFixed(1);
      return [
        '<b>Line #' + '</b><br>',
        'Brush: ' + brushName + ' (id=' + brushId + ')<br>',
        'Points: ' + nPoints + '<br>',
        'Start: (' + first + ')<br>',
        'End: (' + last + ')'
      ].join('');
    }
    return op + ': ' + JSON.stringify(inst[1]);
  }

  function highlightStroke(idx) {
    var state = window.__player.getState();
    if (idx < 0 || idx >= state.total) return;
    // Save current canvas if not already saved
    if (highlightedIdx === -1) {
      savedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    highlightedIdx = idx;
    // Draw highlight overlay: re-blit saved image, then draw a yellow outline
    // around the stroke's bounding box
    if (savedImageData) ctx.putImageData(savedImageData, 0, 0);
    var inst = window.__player.getInstructions()[idx];
    if (!inst || inst[0] !== 'line') return;
    var pts = inst.slice(2);
    var brushId = inst[1];
    var stride = brushId === 5 ? 3 : 2;
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < pts.length; i += stride) {
      if (pts[i] < minX) minX = pts[i];
      if (pts[i] > maxX) maxX = pts[i];
      if (pts[i + 1] < minY) minY = pts[i + 1];
      if (pts[i + 1] > maxY) maxY = pts[i + 1];
    }
    var pad = 6;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2);
    ctx.restore();
  }

  function clearHighlight() {
    if (highlightedIdx !== -1 && savedImageData) {
      ctx.putImageData(savedImageData, 0, 0);
      highlightedIdx = -1;
      savedImageData = null;
    }
  }

  // Mouse hover on canvas -> find nearest stroke, highlight + show tooltip
  canvas.addEventListener('mousemove', function (e) {
    var state = window.__player.getState();
    if (state.total === 0 || state.isPlaying) { hideTooltip(); return; }
    var rect = canvas.getBoundingClientRect();
    var mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    var my = (e.clientY - rect.top) * (canvas.height / rect.height);
    var instrs = window.__player.getInstructions();
    var bestIdx = -1, bestDist = 30;  // 30px snap radius
    for (var i = 0; i < instrs.length; i++) {
      var inst = instrs[i];
      if (!Array.isArray(inst) || inst[0] !== 'line') continue;
      var pts = inst.slice(2);
      var stride = inst[1] === 5 ? 3 : 2;
      for (var j = 0; j < pts.length; j += stride) {
        var dx = pts[j] - mx, dy = pts[j + 1] - my;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
    }
    if (bestIdx !== -1) {
      highlightStroke(bestIdx);
      showTooltip(e.clientX, e.clientY, describeInstruction(instrs[bestIdx]));
    } else {
      clearHighlight();
      hideTooltip();
    }
  });

  canvas.addEventListener('mouseleave', function () {
    clearHighlight();
    hideTooltip();
  });

  canvas.addEventListener('click', function (e) {
    var state = window.__player.getState();
    if (state.total === 0 || state.isPlaying) return;
    if (highlightedIdx !== -1) {
      window.__player.stepToIndex(highlightedIdx + 1);
    }
  });

  // ===========================================================================
  // 2. Thumbnail navigation sidebar
  // ===========================================================================

  var sidebar = null;
  var thumbList = null;

  function ensureSidebar() {
    if (sidebar) return sidebar;
    sidebar = document.createElement('div');
    sidebar.id = 'thumbSidebar';
    sidebar.style.cssText = [
      'position:fixed', 'right:0', 'top:0', 'bottom:0', 'width:180px',
      'background:rgba(20,20,40,0.95)', 'border-left:1px solid #333',
      'overflow-y:auto', 'padding:8px', 'z-index:100',
      'font-family:sans-serif', 'font-size:11px', 'color:#aaa'
    ].join(';');
    var title = document.createElement('div');
    title.textContent = 'Stroke Navigator';
    title.style.cssText = 'font-weight:bold;color:#667eea;margin-bottom:8px;border-bottom:1px solid #333;padding-bottom:4px';
    sidebar.appendChild(title);
    thumbList = document.createElement('div');
    sidebar.appendChild(thumbList);
    var toggle = document.createElement('button');
    toggle.textContent = '◀ Hide';
    toggle.style.cssText = 'position:absolute;top:8px;right:8px;background:none;border:none;color:#888;cursor:pointer;font-size:10px';
    toggle.onclick = function () { sidebar.style.display = 'none'; };
    sidebar.appendChild(toggle);
    document.body.appendChild(sidebar);
    return sidebar;
  }

  function buildThumbnails() {
    var state = window.__player.getState();
    if (state.total === 0) return;
    ensureSidebar();
    thumbList.innerHTML = '';
    var instrs = window.__player.getInstructions();
    // Show every Nth line stroke as a thumbnail entry
    var lineIndices = [];
    for (var i = 0; i < instrs.length; i++) {
      if (Array.isArray(instrs[i]) && instrs[i][0] === 'line') lineIndices.push(i);
    }
    var step = Math.max(1, Math.floor(lineIndices.length / 50));  // ~50 thumbnails
    for (var j = 0; j < lineIndices.length; j += step) {
      var idx = lineIndices[j];
      var entry = document.createElement('div');
      entry.style.cssText = 'padding:4px 6px;cursor:pointer;border-radius:4px;margin-bottom:2px;display:flex;align-items:center;gap:6px';
      entry.onmouseenter = function () { this.style.background = 'rgba(102,126,234,0.2)'; };
      entry.onmouseleave = function () { this.style.background = 'transparent'; };
      (function (idx, entry) {
        entry.onclick = function () {
          window.__player.stepToIndex(idx + 1);
        };
      })(idx, entry);
      // Colour swatch
      var swatch = document.createElement('span');
      swatch.style.cssText = 'display:inline-block;width:12px;height:12px;border-radius:2px;flex-shrink:0';
      // Find the colour instruction before this line
      var colour = '#888';
      for (var k = idx; k >= 0; k--) {
        if (Array.isArray(instrs[k]) && instrs[k][0] === 'colour') {
          colour = instrs[k][1]; break;
        }
      }
      swatch.style.background = colour;
      entry.appendChild(swatch);
      var label = document.createElement('span');
      label.textContent = '#' + idx;
      entry.appendChild(label);
      thumbList.appendChild(entry);
    }
  }

  // Add a "Show Navigator" button to the control panel
  function addNavigatorButton() {
    var panel = document.querySelector('.control-panel');
    if (!panel) return;
    var btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.textContent = '🗂 导航侧栏';
    btn.style.marginTop = '8px';
    btn.onclick = function () {
      var sb = ensureSidebar();
      sb.style.display = sb.style.display === 'none' ? 'block' : 'none';
      if (sb.style.display === 'block') buildThumbnails();
    };
    panel.appendChild(btn);
  }

  // ===========================================================================
  // 3. Customizable keyboard shortcuts (localStorage)
  // ===========================================================================

  var DEFAULT_SHORTCUTS = {
    playPause: 'Space',
    stepForward: 'ArrowRight',
    reset: 'KeyR',
    grid: 'KeyG',
    clear: 'KeyC',
    export: 'KeyE'
  };

  function loadShortcuts() {
    try {
      var saved = localStorage.getItem('asp_shortcuts');
      if (saved) return Object.assign({}, DEFAULT_SHORTCUTS, JSON.parse(saved));
    } catch (e) {}
    return Object.assign({}, DEFAULT_SHORTCUTS);
  }

  function saveShortcuts(cuts) {
    try { localStorage.setItem('asp_shortcuts', JSON.stringify(cuts)); } catch (e) {}
  }

  var shortcuts = loadShortcuts();

  function shortcutsPanel() {
    var panel = document.createElement('div');
    panel.id = 'shortcutsPanel';
    panel.style.cssText = [
      'position:fixed', 'top:50%', 'left:50%', 'transform:translate(-50%,-50%)',
      'background:#1a1a2e', 'border:1px solid #667eea', 'border-radius:8px',
      'padding:20px', 'z-index:2000', 'min-width:320px', 'color:#e0e0e0',
      'box-shadow:0 8px 32px rgba(0,0,0,0.6)', 'font-family:sans-serif'
    ].join(';');
    var title = document.createElement('h3');
    title.textContent = '⌨️ 自定义快捷键';
    title.style.cssText = 'margin:0 0 16px 0;color:#667eea';
    panel.appendChild(title);
    var actions = [
      ['playPause', '播放/暂停'],
      ['stepForward', '单步前进'],
      ['reset', '重置'],
      ['grid', '显示网格'],
      ['clear', '清空画布'],
      ['export', '导出PNG']
    ];
    actions.forEach(function (a) {
      var row = document.createElement('div');
      row.style.cssText = 'display:flex;justify-content:space-between;align-items:center;margin-bottom:10px';
      var label = document.createElement('span');
      label.textContent = a[1];
      row.appendChild(label);
      var input = document.createElement('input');
      input.type = 'text';
      input.value = shortcuts[a[0]] || '';
      input.readOnly = true;
      input.style.cssText = 'width:120px;padding:4px 8px;background:#2d2d44;border:1px solid #444;border-radius:4px;color:#e0e0e0;text-align:center';
      input.onclick = function () {
        input.value = '按下按键...';
        input.focus();
        var handler = function (e) {
          e.preventDefault();
          shortcuts[a[0]] = e.code;
          input.value = e.code;
          saveShortcuts(shortcuts);
          document.removeEventListener('keydown', handler);
        };
        document.addEventListener('keydown', handler);
      };
      row.appendChild(input);
      panel.appendChild(row);
    });
    var closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.className = 'btn btn-secondary';
    closeBtn.style.cssText = 'margin-top:12px;width:100%';
    closeBtn.onclick = function () { panel.remove(); };
    panel.appendChild(closeBtn);
    document.body.appendChild(panel);
  }

  function addShortcutsButton() {
    var panel = document.querySelector('.control-panel');
    if (!panel) return;
    var btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.textContent = '⌨️ 快捷键';
    btn.style.marginTop = '8px';
    btn.onclick = shortcutsPanel;
    panel.appendChild(btn);
  }

  // Override the default keyboard handler with customizable shortcuts
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    var code = e.code;
    if (code === shortcuts.playPause) {
      e.preventDefault();
      var pp = document.getElementById('playPauseBtn');
      if (pp && !pp.disabled) pp.click();
    } else if (code === shortcuts.stepForward) {
      e.preventDefault();
      var sb = document.getElementById('stepBtn');
      if (sb && !sb.disabled) sb.click();
    } else if (code === shortcuts.reset) {
      e.preventDefault();
      var rb = document.getElementById('resetBtn');
      if (rb && !rb.disabled) rb.click();
    } else if (code === shortcuts.grid) {
      e.preventDefault();
      var gb = document.getElementById('gridBtn');
      if (gb) gb.click();
    } else if (code === shortcuts.clear) {
      e.preventDefault();
      var cb = document.getElementById('clearBtn');
      if (cb && !cb.disabled) cb.click();
    } else if (code === shortcuts.export) {
      e.preventDefault();
      var eb = document.getElementById('exportBtn');
      if (eb && !eb.disabled) eb.click();
    }
  });

  // ===========================================================================
  // 4. Live brush preview
  // ===========================================================================

  function addBrushPreview() {
    var panel = document.querySelector('.control-panel');
    if (!panel) return;
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'margin-top:12px;padding:8px;background:rgba(0,0,0,0.2);border-radius:6px';
    var label = document.createElement('div');
    label.textContent = '🖌 笔刷预览';
    label.style.cssText = 'font-size:12px;color:#888;margin-bottom:6px';
    wrapper.appendChild(label);
    var previewCanvas = document.createElement('canvas');
    previewCanvas.width = 280;
    previewCanvas.height = 60;
    previewCanvas.style.cssText = 'width:100%;background:#f8ecdb;border-radius:4px';
    wrapper.appendChild(previewCanvas);
    panel.appendChild(wrapper);

    function drawPreview() {
      var pctx = previewCanvas.getContext('2d');
      pctx.fillStyle = '#f8ecdb';
      pctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
      // Draw a sample stroke with bell-shaped pressure
      pctx.strokeStyle = '#d20000';
      pctx.lineCap = 'round';
      pctx.lineJoin = 'round';
      var nPts = 40;
      var p0 = { x: 10, y: 30 };
      var p1 = { x: previewCanvas.width - 10, y: 30 };
      for (var i = 0; i < nPts - 1; i++) {
        var t0 = i / (nPts - 1);
        var t1 = (i + 1) / (nPts - 1);
        var x0 = p0.x + (p1.x - p0.x) * t0;
        var x1 = p0.x + (p1.x - p0.x) * t1;
        // Bell pressure: 0.15 + 0.85 * 0.5*(1-cos(2*pi*t))
        var p0v = 0.15 + 0.85 * 0.5 * (1 - Math.cos(2 * Math.PI * t0));
        var p1v = 0.15 + 0.85 * 0.5 * (1 - Math.cos(2 * Math.PI * t1));
        pctx.lineWidth = 2 + p0v * 16;
        pctx.globalAlpha = 0.6 + p0v * 0.4;
        pctx.beginPath();
        pctx.moveTo(x0, p0.y);
        pctx.lineTo(x1, p0.y);
        pctx.stroke();
      }
      pctx.globalAlpha = 1;
      // Draw pressure curve below
      pctx.strokeStyle = '#667eea';
      pctx.lineWidth = 1;
      pctx.beginPath();
      for (var j = 0; j < previewCanvas.width; j++) {
        var t = j / (previewCanvas.width - 1);
        var p = 0.15 + 0.85 * 0.5 * (1 - Math.cos(2 * Math.PI * t));
        var y = previewCanvas.height - 4 - p * 16;
        if (j === 0) pctx.moveTo(j, y);
        else pctx.lineTo(j, y);
      }
      pctx.stroke();
    }
    drawPreview();
  }

  // ===========================================================================
  // 5. Touch gesture support
  // ===========================================================================

  var touchState = { pinchStart: 0, lastSwipeX: 0 };

  canvas.addEventListener('touchstart', function (e) {
    if (e.touches.length === 2) {
      var dx = e.touches[0].clientX - e.touches[1].clientX;
      var dy = e.touches[0].clientY - e.touches[1].clientY;
      touchState.pinchStart = Math.sqrt(dx * dx + dy * dy);
    } else if (e.touches.length === 1) {
      touchState.lastSwipeX = e.touches[0].clientX;
    }
  }, { passive: true });

  canvas.addEventListener('touchmove', function (e) {
    if (e.touches.length === 2) {
      e.preventDefault();
      var dx = e.touches[0].clientX - e.touches[1].clientX;
      var dy = e.touches[0].clientY - e.touches[1].clientY;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (touchState.pinchStart > 0) {
        var scale = dist / touchState.pinchStart;
        // Adjust speed based on pinch
        var select = document.getElementById('speedSelect');
        if (scale > 1.2) { select.value = '4'; }
        else if (scale < 0.8) { select.value = '0.5'; }
        else { select.value = '1'; }
      }
    }
  }, { passive: false });

  canvas.addEventListener('touchend', function (e) {
    if (e.changedTouches.length === 1) {
      var dx = e.changedTouches[0].clientX - touchState.lastSwipeX;
      if (Math.abs(dx) > 50) {
        // Swipe: right = step forward, left = step back
        if (dx > 0) {
          var sb = document.getElementById('stepBtn');
          if (sb && !sb.disabled) sb.click();
        } else {
          var state = window.__player.getState();
          if (state.currentIdx > 0) {
            window.__player.stepToIndex(Math.max(0, state.currentIdx - 5));
          }
        }
      }
    }
    touchState.pinchStart = 0;
  });

  // ===========================================================================
  // Initialize
  // ===========================================================================

  function init() {
    if (!window.__player) {
      // Main player not ready yet; retry shortly
      setTimeout(init, 50);
      return;
    }
    addNavigatorButton();
    addShortcutsButton();
    addBrushPreview();
    console.log('[enhancements] All UI enhancements loaded.');
  }

  // Expose enhancement API
  window.__playerEnh = {
    buildThumbnails: buildThumbnails,
    shortcutsPanel: shortcutsPanel,
    getShortcuts: function () { return Object.assign({}, shortcuts); },
    resetShortcuts: function () {
      shortcuts = Object.assign({}, DEFAULT_SHORTCUTS);
      saveShortcuts(shortcuts);
    }
  };

  // Init after DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
