"""Streamlit Components v2 live webcam widget (non-iframed, native getUserMedia + YOLO overlay)."""

from __future__ import annotations

import streamlit as st

_LIVE_WEBCAM_HTML = """
<div class="do-live-root">
  <div class="do-bar">
    <button id="do-start-btn" class="do-btn do-btn-primary" type="button">Start Camera</button>
    <button id="do-pause-btn" class="do-btn" type="button" disabled>Pause</button>
    <button id="do-shot-btn" class="do-btn" type="button" disabled>Capture</button>
    <select id="do-cam-select" class="do-select" title="Select camera">
      <option value="">Default Camera</option>
    </select>
    <button id="do-opts-btn" class="do-btn" type="button">Options</button>
    <div class="do-badge">
      <span id="do-dot" class="do-dot"></span>
      <span id="do-status-text">Idle</span>
    </div>
  </div>

  <div id="do-opts-panel" class="do-opts-panel">
    <div class="do-opt-item">
      <label>Confidence <span id="do-conf-val">0.35</span></label>
      <input id="do-conf" type="range" min="0.10" max="0.90" step="0.05" value="0.35" />
    </div>
    <div class="do-opt-item">
      <label>Resolution</label>
      <select id="do-imgsz" class="do-select do-select-full">
        <option value="256">256 px (Fastest)</option>
        <option value="320" selected>320 px (Balanced)</option>
        <option value="416">416 px</option>
        <option value="480">480 px (Sharp)</option>
      </select>
    </div>
    <div class="do-opt-item">
      <label>Tracker</label>
      <select id="do-tracker" class="do-select do-select-full">
        <option value="bytetrack">ByteTrack</option>
        <option value="botsort">BoT-SORT</option>
      </select>
    </div>
    <div class="do-opt-checks">
      <label><input id="do-track-on" type="checkbox" checked /> Tracking</label>
      <label><input id="do-trails-on" type="checkbox" checked /> Trails</label>
      <label><input id="do-line-on" type="checkbox" /> Counting Line</label>
      <label><input id="do-roi-on" type="checkbox" /> ROI Zone</label>
      <button id="do-reset-btn" class="do-btn do-btn-sm" type="button">Reset Counts</button>
    </div>
  </div>

  <div id="do-error" class="do-error"></div>

  <div class="do-grid">
    <div>
      <div id="do-stage" class="do-stage">
        <video id="do-video" autoplay playsinline muted></video>
        <canvas id="do-canvas"></canvas>
        <div id="do-placeholder" class="do-placeholder">
          <h3>Camera is off</h3>
          <p>Click <b>Start Camera</b> to stream live video directly from your browser with real-time AI object detection.</p>
        </div>
      </div>
      <div id="do-toast" class="do-toast"></div>
    </div>

    <div class="do-side">
      <div class="do-metrics">
        <div class="do-metric">
          <div class="do-metric-label">AI FPS</div>
          <div id="do-m-fps" class="do-metric-val">—</div>
        </div>
        <div class="do-metric">
          <div class="do-metric-label">In Frame</div>
          <div id="do-m-frame" class="do-metric-val">0</div>
        </div>
        <div class="do-metric">
          <div class="do-metric-label">Unique IDs</div>
          <div id="do-m-unique" class="do-metric-val">0</div>
        </div>
        <div class="do-metric">
          <div class="do-metric-label">Latency</div>
          <div id="do-m-lat" class="do-metric-val">—</div>
        </div>
      </div>

      <div id="do-zone-metrics" class="do-metrics" style="display:none;">
        <div class="do-metric">
          <div class="do-metric-label">Line In / Out</div>
          <div id="do-m-line" class="do-metric-val">0 / 0</div>
        </div>
        <div class="do-metric">
          <div class="do-metric-label">ROI Now / Uniq</div>
          <div id="do-m-roi" class="do-metric-val">0 / 0</div>
        </div>
      </div>

      <div class="do-classes-box">
        <div class="do-classes-title">Detected Classes</div>
        <div id="do-pills" class="do-pills">
          <span class="do-muted">Waiting for camera...</span>
        </div>
      </div>
    </div>
  </div>
</div>
"""

_LIVE_WEBCAM_CSS = """
.do-live-root {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  color: #0f172a;
  width: 100%;
  box-sizing: border-box;
}
.do-live-root * { box-sizing: border-box; }
.do-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}
.do-btn {
  appearance: none;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #0f172a;
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, opacity 0.15s;
}
.do-btn:hover:not(:disabled) { background: #f1f5f9; }
.do-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.do-btn-primary {
  background: #2563eb;
  color: #ffffff;
  border-color: #2563eb;
}
.do-btn-primary:hover:not(:disabled) { background: #1d4ed8; border-color: #1d4ed8; }
.do-btn-danger {
  background: #dc2626;
  color: #ffffff;
  border-color: #dc2626;
}
.do-btn-danger:hover:not(:disabled) { background: #b91c1c; border-color: #b91c1c; }
.do-btn-sm {
  padding: 4px 10px;
  font-size: 12px;
  margin-left: auto;
}
.do-select {
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #0f172a;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 13px;
  min-width: 185px;
  max-width: 260px;
}
.do-select-full { min-width: 0; width: 100%; }
.do-badge {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #475569;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
}
.do-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #94a3b8;
}
.do-dot.live {
  background: #16a34a;
  box-shadow: 0 0 8px #16a34a;
}
.do-opts-panel {
  display: none;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  font-size: 12.5px;
}
.do-opts-panel.open { display: grid; }
.do-opt-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.do-opt-item label {
  color: #475569;
  font-size: 12px;
  display: flex;
  justify-content: space-between;
}
.do-opt-checks {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  grid-column: 1 / -1;
  padding-top: 8px;
  border-top: 1px solid #e2e8f0;
}
.do-opt-checks label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12.5px;
  color: #0f172a;
  cursor: pointer;
}
.do-grid {
  display: grid;
  grid-template-columns: minmax(0, 2.2fr) minmax(240px, 1fr);
  gap: 14px;
  align-items: start;
}
@media (max-width: 820px) {
  .do-grid { grid-template-columns: 1fr; }
}
.do-stage {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.do-stage.active {
  background: #0f172a;
  border-color: #cbd5e1;
}
.do-stage video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: none;
}
.do-stage canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  display: none;
}
.do-placeholder {
  text-align: center;
  padding: 24px;
  color: #475569;
  max-width: 400px;
}
.do-placeholder h3 {
  margin: 0 0 6px 0;
  font-size: 15px;
  font-weight: 600;
  color: #0f172a;
}
.do-placeholder p {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  color: #64748b;
}
.do-error {
  display: none;
  margin-bottom: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #fca5a5;
  background: #fef2f2;
  color: #991b1b;
  font-size: 13px;
  line-height: 1.45;
}
.do-toast {
  display: none;
  margin-top: 8px;
  font-size: 12.5px;
  color: #2563eb;
  font-weight: 500;
}
.do-side {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.do-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.do-metric {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 12px;
}
.do-metric-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.do-metric-val {
  font-size: 20px;
  font-weight: 600;
  margin-top: 2px;
  color: #0f172a;
}
.do-classes-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 92px;
}
.do-classes-title {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
}
.do-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.do-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  font-size: 12px;
  color: #0f172a;
}
.do-pill b { color: #2563eb; }
.do-muted { font-size: 12px; color: #64748b; }
"""

_LIVE_WEBCAM_JS = """
export default function(component) {
  const { data, parentElement } = component;
  const CFG = data || {};
  const q = (sel) => parentElement.querySelector(sel);

  const root = q(".do-live-root");
  if (!root) return;

  // Store mutable config so re-renders never destroy an active webcam stream
  if (root._doUpdateConfig) {
    root._doUpdateConfig(CFG);
    return;
  }

  function resolveApiBase(base) {
    if (!base) return "";
    try {
      const u = new URL(base);
      if (window.location.hostname && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
        return `${window.location.protocol}//${window.location.hostname}:${u.port || 8511}`;
      }
    } catch (_) {}
    return base;
  }

  let apiBase = resolveApiBase(CFG.apiBase || "");
  let defaultIou = CFG.iou ?? 0.45;

  const stage = q("#do-stage");
  const video = q("#do-video");
  const canvas = q("#do-canvas");
  const ctx = canvas.getContext("2d");
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  const startBtn = q("#do-start-btn");
  const pauseBtn = q("#do-pause-btn");
  const shotBtn = q("#do-shot-btn");
  const optsBtn = q("#do-opts-btn");
  const optsPanel = q("#do-opts-panel");
  const resetBtn = q("#do-reset-btn");
  const camSelect = q("#do-cam-select");
  const placeholder = q("#do-placeholder");
  const errBox = q("#do-error");
  const toastEl = q("#do-toast");
  const dotEl = q("#do-dot");
  const statusEl = q("#do-status-text");

  const confInput = q("#do-conf");
  const confVal = q("#do-conf-val");
  const imgszSelect = q("#do-imgsz");
  const trackerSelect = q("#do-tracker");
  const trackCheck = q("#do-track-on");
  const trailsCheck = q("#do-trails-on");
  const lineCheck = q("#do-line-on");
  const roiCheck = q("#do-roi-on");

  const mFps = q("#do-m-fps");
  const mFrame = q("#do-m-frame");
  const mUnique = q("#do-m-unique");
  const mLat = q("#do-m-lat");
  const zoneMetrics = q("#do-zone-metrics");
  const mLine = q("#do-m-line");
  const mRoi = q("#do-m-roi");
  const pillsEl = q("#do-pills");

  confInput.value = String(CFG.conf ?? 0.35);
  confVal.textContent = Number(confInput.value).toFixed(2);
  imgszSelect.value = String(CFG.imgsz ?? 320);
  trackerSelect.value = String(CFG.tracker ?? "bytetrack");

  root._doUpdateConfig = (nextCfg) => {
    if (nextCfg && nextCfg.apiBase) apiBase = resolveApiBase(nextCfg.apiBase);
    if (nextCfg && nextCfg.iou !== undefined) defaultIou = nextCfg.iou;
  };

  const COLORS = [
    "#2563eb", "#16a34a", "#ea580c", "#ca8a04",
    "#db2777", "#0d9488", "#9333ea", "#dc2626"
  ];

  let stream = null;
  let running = false;
  let paused = false;
  let inFlight = false;
  let loopTimer = null;
  let lastData = null;
  const fpsTimes = [];

  function showError(msg) {
    if (!msg) {
      errBox.style.display = "none";
      errBox.textContent = "";
      return;
    }
    errBox.style.display = "block";
    errBox.textContent = msg;
  }

  function showToast(msg) {
    toastEl.textContent = msg;
    toastEl.style.display = "block";
    setTimeout(() => {
      if (toastEl.textContent === msg) toastEl.style.display = "none";
    }, 3500);
  }

  function setStatus(label, isLive) {
    statusEl.textContent = label;
    dotEl.classList.toggle("live", !!isLive);
  }

  async function refreshCameraList() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      const prev = camSelect.value;
      camSelect.innerHTML = "";
      if (cams.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No camera found";
        camSelect.appendChild(opt);
        return;
      }
      cams.forEach((cam, idx) => {
        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.textContent = cam.label || `Camera ${idx + 1}`;
        camSelect.appendChild(opt);
      });
      if (prev && cams.some((c) => c.deviceId === prev)) {
        camSelect.value = prev;
      }
    } catch (_) {}
  }

  function drawOverlay(payload) {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }
    ctx.clearRect(0, 0, vw, vh);

    if (roiCheck.checked) {
      const rx = 0.15 * vw, ry = 0.15 * vh, rw = 0.70 * vw, rh = 0.70 * vh;
      ctx.fillStyle = "rgba(37, 99, 235, 0.12)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);
    }

    if (lineCheck.checked) {
      const ly = 0.5 * vh;
      ctx.beginPath();
      ctx.moveTo(0, ly);
      ctx.lineTo(vw, ly);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (!payload) return;

    if (trailsCheck.checked && payload.trails) {
      Object.entries(payload.trails).forEach(([tid, pts]) => {
        if (!pts || pts.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = COLORS[Number(tid) % COLORS.length];
        ctx.lineWidth = 2;
        pts.forEach((pt, idx) => {
          const px = pt[0] * vw, py = pt[1] * vh;
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      });
    }

    (payload.detections || []).forEach((det) => {
      const x1 = det.x1 * vw, y1 = det.y1 * vh;
      const w = (det.x2 - det.x1) * vw, h = (det.y2 - det.y1) * vh;
      const color = COLORS[Math.abs(det.class_id) % COLORS.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, w, h);

      const idPart = det.track_id !== null && det.track_id !== undefined ? `#${det.track_id} ` : "";
      const label = `${idPart}${det.class_name} ${Math.round(det.confidence * 100)}%`;
      ctx.font = "600 13px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
      const tw = ctx.measureText(label).width;
      const th = 20;
      const ly = y1 - th >= 0 ? y1 - th : y1;

      ctx.fillStyle = color;
      ctx.fillRect(x1, ly, tw + 10, th);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 5, ly + 14);
    });
  }

  function updateMetrics(payload) {
    const now = performance.now();
    fpsTimes.push(now);
    while (fpsTimes.length > 20) fpsTimes.shift();
    let fps = 0;
    if (fpsTimes.length >= 2) {
      const dt = (fpsTimes[fpsTimes.length - 1] - fpsTimes[0]) / 1000;
      if (dt > 0) fps = (fpsTimes.length - 1) / dt;
    }

    const counts = payload.counts || {};
    mFps.textContent = fps > 0 ? fps.toFixed(1) : "—";
    mFrame.textContent = String(counts.frame_total ?? 0);
    mUnique.textContent = String(counts.unique_tracked_total ?? 0);
    mLat.textContent = `${Math.round(payload.inference_ms || 0)} ms`;

    const showZones = lineCheck.checked || roiCheck.checked;
    zoneMetrics.style.display = showZones ? "grid" : "none";
    if (showZones) {
      mLine.textContent = `${counts.line_in_count ?? 0} / ${counts.line_out_count ?? 0}`;
      mRoi.textContent = `${counts.roi_current_count ?? 0} / ${counts.roi_unique_total ?? 0}`;
    }

    const entries = Object.entries(counts.frame_class_counts || {});
    if (entries.length === 0) {
      pillsEl.innerHTML = `<span class="do-muted">No objects in frame</span>`;
    } else {
      pillsEl.innerHTML = entries
        .map(([k, v]) => `<span class="do-pill">${k} <b>${v}</b></span>`)
        .join("");
    }
  }

  async function inferFrame() {
    if (!running || paused || inFlight) return;
    if (video.readyState < 2 || !video.videoWidth) {
      loopTimer = setTimeout(inferFrame, 60);
      return;
    }

    inFlight = true;
    try {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const scale = Math.min(1, 640 / vw);
      captureCanvas.width = Math.round(vw * scale);
      captureCanvas.height = Math.round(vh * scale);
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.72));
      if (!blob || !running) return;

      const qs = new URLSearchParams({
        conf: confInput.value,
        iou: String(defaultIou),
        imgsz: imgszSelect.value,
        track: trackCheck.checked ? "1" : "0",
        tracker: trackerSelect.value,
        line: lineCheck.checked ? "1" : "0",
        roi: roiCheck.checked ? "1" : "0",
      });

      const resp = await fetch(`${apiBase}/detect?${qs.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      if (!resp.ok) throw new Error(`Inference API HTTP ${resp.status}`);
      const payload = await resp.json();
      if (running && !paused) {
        lastData = payload;
        drawOverlay(payload);
        updateMetrics(payload);
      }
    } catch (err) {
      if (running) showError(`Detection error: ${err.message}`);
    } finally {
      inFlight = false;
      if (running && !paused) {
        loopTimer = setTimeout(inferFrame, 15);
      }
    }
  }

  async function startCamera(deviceId) {
    showError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Browser camera API is unavailable. Open DetectObject via http://localhost:8501 in Chrome or Edge.");
      return;
    }

    stopCamera(true);
    try {
      const videoConstraints = deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
        : { width: { ideal: 640 }, height: { ideal: 480 } };

      stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
      video.srcObject = stream;
      await video.play();

      running = true;
      paused = false;
      fpsTimes.length = 0;

      await fetch(`${apiBase}/reset`, { method: "POST" }).catch(() => {});
      await refreshCameraList();

      const activeTrack = stream.getVideoTracks()[0];
      if (activeTrack && activeTrack.getSettings().deviceId) {
        camSelect.value = activeTrack.getSettings().deviceId;
      }

      stage.classList.add("active");
      placeholder.style.display = "none";
      video.style.display = "block";
      canvas.style.display = "block";

      startBtn.textContent = "Stop Camera";
      startBtn.classList.remove("do-btn-primary");
      startBtn.classList.add("do-btn-danger");
      pauseBtn.disabled = false;
      pauseBtn.textContent = "Pause";
      shotBtn.disabled = false;
      setStatus("Live", true);

      inferFrame();
    } catch (err) {
      stopCamera(true);
      const name = err && err.name ? err.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        showError("Camera permission denied. Click the camera/lock icon in your browser address bar, allow Camera access, and click Start Camera again.");
      } else if (name === "NotReadableError" || name === "TrackStartError") {
        showError("Camera is in use by another app or tab. Close other apps using the webcam and click Start Camera again.");
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        showError("No webcam detected on this device. Connect a camera and try again.");
      } else {
        showError(`Could not connect to camera: ${err.message || err}`);
      }
    }
  }

  function stopCamera(silent) {
    running = false;
    paused = false;
    inFlight = false;
    if (loopTimer) {
      clearTimeout(loopTimer);
      loopTimer = null;
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
    video.style.display = "none";
    canvas.style.display = "none";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stage.classList.remove("active");
    placeholder.style.display = "block";

    startBtn.textContent = "Start Camera";
    startBtn.classList.remove("do-btn-danger");
    startBtn.classList.add("do-btn-primary");
    pauseBtn.disabled = true;
    pauseBtn.textContent = "Pause";
    shotBtn.disabled = true;
    if (!silent) setStatus("Idle", false);
  }

  startBtn.addEventListener("click", () => {
    if (running) stopCamera(false);
    else startCamera(camSelect.value || null);
  });

  pauseBtn.addEventListener("click", () => {
    if (!running) return;
    paused = !paused;
    if (paused) {
      video.pause();
      pauseBtn.textContent = "Resume";
      setStatus("Paused", false);
    } else {
      video.play();
      pauseBtn.textContent = "Pause";
      setStatus("Live", true);
      inferFrame();
    }
  });

  camSelect.addEventListener("change", () => {
    if (running) startCamera(camSelect.value || null);
  });

  optsBtn.addEventListener("click", () => {
    optsPanel.classList.toggle("open");
  });

  confInput.addEventListener("input", () => {
    confVal.textContent = Number(confInput.value).toFixed(2);
  });

  resetBtn.addEventListener("click", async () => {
    await fetch(`${apiBase}/reset`, { method: "POST" }).catch(() => {});
    mUnique.textContent = "0";
    mLine.textContent = "0 / 0";
    mRoi.textContent = "0 / 0";
    showToast("Counts reset.");
  });

  shotBtn.addEventListener("click", async () => {
    if (!running || !video.videoWidth) return;
    const outCanvas = document.createElement("canvas");
    outCanvas.width = video.videoWidth;
    outCanvas.height = video.videoHeight;
    const octx = outCanvas.getContext("2d");
    octx.drawImage(video, 0, 0, outCanvas.width, outCanvas.height);
    octx.drawImage(canvas, 0, 0, outCanvas.width, outCanvas.height);
    const dataUrl = outCanvas.toDataURL("image/png");

    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `detectobject_webcam_${Date.now()}.png`;
    a.click();

    try {
      const resp = await fetch(`${apiBase}/save_screenshot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_data_url: dataUrl,
          counts: lastData ? lastData.counts : {},
          detections: lastData ? lastData.detections : [],
        }),
      });
      if (resp.ok) {
        const saved = await resp.json();
        showToast(`Saved outputs/screenshots/${saved.filename}`);
      }
    } catch (_) {}
  });

  refreshCameraList();

  // Official Streamlit Components v2 unmount cleanup hook
  return () => {
    stopCamera(true);
  };
}
"""

_live_webcam_component = st.components.v2.component(
    "detectobject_live_webcam",
    html=_LIVE_WEBCAM_HTML,
    css=_LIVE_WEBCAM_CSS,
    js=_LIVE_WEBCAM_JS,
    isolate_styles=True,
)


def render_live_webcam(
    api_base_url: str,
    default_conf: float = 0.35,
    default_iou: float = 0.45,
    default_imgsz: int = 320,
    default_tracker: str = "bytetrack",
) -> None:
    """Mount the live webcam Streamlit v2 component."""
    _live_webcam_component(
        key="detectobject_live_webcam_instance",
        data={
            "apiBase": api_base_url,
            "conf": round(float(default_conf), 2),
            "iou": round(float(default_iou), 2),
            "imgsz": int(default_imgsz),
            "tracker": default_tracker,
        },
    )
