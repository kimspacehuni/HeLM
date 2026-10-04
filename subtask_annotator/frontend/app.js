"use strict";

const state = {
  info: null,
  episode: null,
  frame: 0,
  cams: [],
  labels: [],
  pendingStart: null,
  playing: false,
  playRangeEnd: null,
  videoMode: false,
  videoSrcEp: null,
  _timeUpdateHandler: null,
  _endedHandler: null,
  loadSeq: 0,
  editingLabels: false,
};

const els = {
  episodeSelect: document.getElementById("episodeSelect"),
  episodeMeta: document.getElementById("episodeMeta"),
  annotator: document.getElementById("annotatorInput"),
  camLeft: document.getElementById("camLeft"),
  camRight: document.getElementById("camRight"),
  videoLeft: document.getElementById("videoLeft"),
  videoRight: document.getElementById("videoRight"),
  camLeftLabel: document.getElementById("camLeftLabel"),
  camRightLabel: document.getElementById("camRightLabel"),
  timeline: document.getElementById("timeline"),
  frameInput: document.getElementById("frameInput"),
  frameTotal: document.getElementById("frameTotal"),
  btnPrev: document.getElementById("btnPrev"),
  btnNext: document.getElementById("btnNext"),
  btnPrev10: document.getElementById("btnPrev10"),
  btnNext10: document.getElementById("btnNext10"),
  btnPlay: document.getElementById("btnPlay"),
  pendingRange: document.getElementById("pendingRange"),
  subtaskList: document.getElementById("subtaskList"),
  subtaskBars: document.getElementById("subtaskBars"),
  labelButtons: document.getElementById("labelButtons"),
  btnEditLabels: document.getElementById("btnEditLabels"),
  labelEditor: document.getElementById("labelEditor"),
  labelEditList: document.getElementById("labelEditList"),
  labelAddForm: document.getElementById("labelAddForm"),
  labelAddInput: document.getElementById("labelAddInput"),
  btnCloseEdit: document.getElementById("btnCloseEdit"),
  toast: document.getElementById("toast"),
};

function showToast(msg, isError = false) {
  els.toast.textContent = msg;
  els.toast.classList.toggle("error", isError);
  els.toast.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { els.toast.hidden = true; }, 2500);
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function frameUrl(cam, frame) {
  const ep = state.episode.episode_index;
  return `/api/frame?ep=${ep}&cam=${encodeURIComponent(cam)}&frame=${frame}`;
}

function setFrame(frame, { scrub = false, fromVideo = false } = {}) {
  if (!state.episode) return;
  if (state.videoMode && !fromVideo) exitVideoMode({ skipRefresh: true });
  const max = state.episode.length - 1;
  frame = Math.max(0, Math.min(max, frame | 0));
  state.frame = frame;
  els.timeline.value = String(frame);
  els.frameInput.value = String(frame);
  updatePendingBar();
  updatePlayhead();

  if (state.videoMode) return;

  const seq = ++state.loadSeq;
  for (const [img, cam] of [[els.camLeft, state.cams[0]], [els.camRight, state.cams[1]]]) {
    if (!cam) continue;
    const url = frameUrl(cam, frame);
    if (scrub) {
      img.src = url;
    } else {
      const pre = new Image();
      pre.onload = () => { if (seq === state.loadSeq) img.src = url; };
      pre.onerror = () => { if (seq === state.loadSeq) img.src = url; };
      pre.src = url;
    }
  }
}

function stepFrame(delta) {
  setFrame(state.frame + delta);
}

function videoSrcUrl(cam) {
  return `/api/video?ep=${state.episode.episode_index}&cam=${encodeURIComponent(cam)}`;
}

function enterVideoMode(startFrame, endFrame) {
  if (!state.episode) return;
  const fps = state.episode.fps;
  const max = state.episode.length - 1;
  startFrame = Math.max(0, Math.min(max, startFrame | 0));
  const pairs = [[els.videoLeft, state.cams[0]], [els.videoRight, state.cams[1]]];

  for (const [v, cam] of pairs) {
    if (!cam) { v.hidden = true; continue; }
    const url = videoSrcUrl(cam);
    if (state.videoSrcEp !== state.episode.episode_index || !v.src) {
      v.src = url;
      v.load();
    }
    v.hidden = false;
  }
  state.videoSrcEp = state.episode.episode_index;
  els.camLeft.style.display = "none";
  els.camRight.style.display = "none";

  const primary = els.videoLeft;
  const secondary = els.videoRight;

  const seek = () => {
    primary.currentTime = startFrame / fps;
    secondary.currentTime = startFrame / fps;
  };
  if (primary.readyState >= 1) seek();
  else primary.addEventListener("loadedmetadata", seek, { once: true });

  const onTimeUpdate = () => {
    const cur = primary.currentTime;
    const curFrame = Math.floor(cur * fps);
    state.frame = Math.min(max, Math.max(0, curFrame));
    els.timeline.value = String(state.frame);
    els.frameInput.value = String(state.frame);
    updatePlayhead();
    updatePendingBar();
    if (secondary && !secondary.hidden && Math.abs(secondary.currentTime - cur) > 0.15) {
      secondary.currentTime = cur;
    }
    if (endFrame != null && cur >= endFrame / fps) {
      exitVideoMode();
    }
  };
  const onEnded = () => exitVideoMode();

  primary.addEventListener("timeupdate", onTimeUpdate);
  primary.addEventListener("ended", onEnded);
  state._timeUpdateHandler = onTimeUpdate;
  state._endedHandler = onEnded;

  state.videoMode = true;
  state.playing = true;
  state.playRangeEnd = endFrame;
  els.btnPlay.textContent = "\u23F8";

  Promise.all([primary.play(), secondary.play().catch(() => {})])
    .catch(err => {
      console.error("video play failed", err);
      exitVideoMode();
    });
}

function exitVideoMode({ skipRefresh = false } = {}) {
  if (!state.videoMode) return;
  const primary = els.videoLeft;
  const secondary = els.videoRight;
  if (state._timeUpdateHandler) {
    primary.removeEventListener("timeupdate", state._timeUpdateHandler);
    state._timeUpdateHandler = null;
  }
  if (state._endedHandler) {
    primary.removeEventListener("ended", state._endedHandler);
    state._endedHandler = null;
  }
  const fps = state.episode.fps;
  const cur = primary.currentTime;
  primary.pause();
  secondary.pause();

  state.videoMode = false;
  state.playing = false;
  state.playRangeEnd = null;
  els.btnPlay.textContent = "\u25B6";

  els.videoLeft.hidden = true;
  els.videoRight.hidden = true;
  els.camLeft.style.display = "";
  els.camRight.style.display = "";

  const finalFrame = Math.min(state.episode.length - 1, Math.max(0, Math.floor(cur * fps)));
  if (!skipRefresh) setFrame(finalFrame);
  else { state.frame = finalFrame; els.timeline.value = String(finalFrame); els.frameInput.value = String(finalFrame); }
}

function togglePlay() {
  if (state.videoMode) { exitVideoMode(); return; }
  enterVideoMode(state.frame, null);
}

function playRange(startFrame, endFrame) {
  if (state.videoMode) exitVideoMode({ skipRefresh: true });
  enterVideoMode(startFrame, endFrame);
}

function updatePendingBar() {
  if (state.pendingStart == null) {
    els.pendingRange.textContent = "";
    return;
  }
  const s = state.pendingStart;
  const c = state.frame;
  const lo = Math.min(s, c), hi = Math.max(s, c);
  els.pendingRange.textContent = `pending: ${s} \u2192 ${c}  (${hi - lo + 1} frames)`;
}

function renderLabels() {
  els.labelButtons.innerHTML = "";
  if (state.labels.length === 0) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent = "No labels. Click 'edit' to add.";
    els.labelButtons.appendChild(hint);
    return;
  }
  state.labels.forEach((lbl, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "label-btn";
    if (state.pendingStart != null) btn.classList.add("armed");
    const hot = document.createElement("span");
    hot.className = "hotkey";
    hot.textContent = i < 9 ? String(i + 1) : "";
    const text = document.createElement("span");
    text.textContent = lbl;
    btn.append(hot, text);
    btn.addEventListener("click", () => saveWithLabel(lbl));
    els.labelButtons.appendChild(btn);
  });
}

function renderLabelEditor() {
  els.labelEditList.innerHTML = "";
  state.labels.forEach((lbl, i) => {
    const li = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = lbl;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "del";
    del.textContent = "remove";
    del.addEventListener("click", async () => {
      const next = state.labels.slice();
      next.splice(i, 1);
      await saveLabels(next);
    });
    li.append(text, del);
    els.labelEditList.appendChild(li);
  });
}

async function saveLabels(labels) {
  try {
    const res = await fetchJSON("/api/labels", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    });
    state.labels = res.labels;
    renderLabels();
    renderLabelEditor();
  } catch (e) {
    showToast(e.message, true);
  }
}

async function saveWithLabel(label) {
  if (state.pendingStart == null) {
    showToast("press [ to mark start first", true);
    return;
  }
  const s = state.pendingStart;
  const c = state.frame;
  const [lo, hi] = s <= c ? [s, c] : [c, s];
  try {
    await fetchJSON("/api/subtasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episode_index: state.episode.episode_index,
        start_frame: lo,
        end_frame: hi,
        label,
        annotator: els.annotator.value.trim() || null,
      }),
    });
    state.pendingStart = null;
    updatePendingBar();
    renderLabels();
    await refreshEpisode();
    showToast(`saved: ${label} [${lo}\u2013${hi}]`);
  } catch (e) {
    showToast(e.message, true);
  }
}

function labelColor(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  return `hsl(${hue}, 55%, 65%)`;
}

function renderSubtaskBars() {
  els.subtaskBars.innerHTML = "";
  if (!state.episode) return;
  const len = state.episode.length;
  const denom = Math.max(1, len - 1);
  const subs = state.episode.subtasks || [];
  for (const s of subs) {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.left = `${(s.start_frame / denom) * 100}%`;
    const widthPct = ((s.end_frame - s.start_frame + 1) / len) * 100;
    bar.style.width = `${Math.max(0.3, widthPct)}%`;
    bar.style.background = labelColor(s.label);
    bar.title = `${s.label} [${s.start_frame}\u2013${s.end_frame}]`;
    bar.textContent = s.label;
    bar.dataset.id = s.id;
    bar.addEventListener("click", () => playRange(s.start_frame, s.end_frame));
    els.subtaskBars.appendChild(bar);
  }
  const ph = document.createElement("div");
  ph.className = "playhead";
  ph.id = "playhead";
  ph.style.left = `${(state.frame / denom) * 100}%`;
  els.subtaskBars.appendChild(ph);
  updateActiveBar();
}

function updatePlayhead() {
  const ph = document.getElementById("playhead");
  if (!ph || !state.episode) return;
  const denom = Math.max(1, state.episode.length - 1);
  ph.style.left = `${(state.frame / denom) * 100}%`;
  updateActiveBar();
}

function updateActiveBar() {
  const subs = state.episode?.subtasks || [];
  const byId = new Map(subs.map(s => [s.id, s]));
  for (const bar of els.subtaskBars.querySelectorAll(".bar")) {
    const s = byId.get(bar.dataset.id);
    if (!s) continue;
    bar.classList.toggle("active", state.frame >= s.start_frame && state.frame <= s.end_frame);
  }
}

function renderSubtasks() {
  els.subtaskList.innerHTML = "";
  const subs = [...(state.episode?.subtasks || [])].sort((a, b) => a.start_frame - b.start_frame);
  if (subs.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No subtasks yet.";
    li.style.color = "#777";
    li.style.fontStyle = "italic";
    els.subtaskList.appendChild(li);
    return;
  }
  for (const s of subs) {
    const li = document.createElement("li");

    const row1 = document.createElement("div");
    row1.className = "row1";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = s.label;
    const range = document.createElement("span");
    range.className = "range";
    range.textContent = `${s.start_frame}\u2013${s.end_frame}`;
    row1.append(label, range);

    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";
    const left = document.createElement("div");
    left.style.display = "flex"; left.style.gap = "0.6rem";
    const seekStart = document.createElement("button");
    seekStart.className = "seek"; seekStart.textContent = "\u21E4 start";
    seekStart.addEventListener("click", () => setFrame(s.start_frame));
    const seekEnd = document.createElement("button");
    seekEnd.className = "seek"; seekEnd.textContent = "end \u21E5";
    seekEnd.addEventListener("click", () => setFrame(s.end_frame));
    const play = document.createElement("button");
    play.className = "seek"; play.textContent = "\u25B6 play";
    play.addEventListener("click", () => playRange(s.start_frame, s.end_frame));
    left.append(seekStart, seekEnd, play);

    const del = document.createElement("button");
    del.className = "del"; del.textContent = "delete";
    del.addEventListener("click", async () => {
      if (!confirm(`Delete subtask "${s.label}" (${s.start_frame}\u2013${s.end_frame})?`)) return;
      try {
        await fetchJSON(`/api/subtasks/${encodeURIComponent(s.id)}`, { method: "DELETE" });
        await refreshEpisode();
        showToast("deleted");
      } catch (e) { showToast(e.message, true); }
    });

    const annotator = document.createElement("span");
    annotator.textContent = s.annotator || "";
    metaRow.append(left, annotator, del);

    li.append(row1, metaRow);
    els.subtaskList.appendChild(li);
  }
}

async function refreshEpisode() {
  const ep = state.episode.episode_index;
  const data = await fetchJSON(`/api/episodes/${ep}`);
  state.episode = data;
  renderSubtasks();
  renderSubtaskBars();
}

async function loadEpisode(ep) {
  if (state.playing) togglePlay();
  state.pendingStart = null;
  updatePendingBar();

  const data = await fetchJSON(`/api/episodes/${ep}`);
  state.episode = data;
  state.cams = data.video_keys.slice(0, 2);
  els.camLeftLabel.textContent = state.cams[0] || "";
  els.camRightLabel.textContent = state.cams[1] || "";

  els.timeline.max = String(data.length - 1);
  els.frameInput.max = String(data.length - 1);
  els.frameTotal.textContent = `/ ${data.length - 1}`;
  els.episodeMeta.textContent =
    `${data.length} frames @ ${data.fps} FPS \u00B7 "${(data.tasks[0] || "").slice(0, 60)}"`;

  renderSubtasks();
  renderSubtaskBars();
  renderLabels();
  setFrame(0);
}

function bindKeyboard() {
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "textarea" || tag === "select";
    if (typing) return;

    if (e.key === "ArrowLeft") { e.preventDefault(); stepFrame(e.shiftKey ? -10 : -1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); stepFrame(e.shiftKey ? 10 : 1); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "[") {
      state.pendingStart = state.frame;
      updatePendingBar();
      renderLabels();
      showToast(`start: ${state.frame}`);
    }
    else if (e.key === "Escape") {
      if (state.pendingStart != null) {
        state.pendingStart = null;
        updatePendingBar();
        renderLabels();
        showToast("pending cleared");
      }
      if (state.editingLabels) toggleLabelEditor(false);
    }
    else if (/^[1-9]$/.test(e.key)) {
      const idx = parseInt(e.key, 10) - 1;
      if (idx < state.labels.length) {
        e.preventDefault();
        saveWithLabel(state.labels[idx]);
      }
    }
  });
}

function toggleLabelEditor(on) {
  state.editingLabels = on;
  els.labelEditor.hidden = !on;
  els.btnEditLabels.textContent = on ? "close" : "edit";
  if (on) {
    renderLabelEditor();
    els.labelAddInput.focus();
  }
}

function bindScroll() {
  const area = document.querySelector(".cams");
  area.addEventListener("wheel", (e) => {
    if (!state.episode) return;
    e.preventDefault();
    const dir = e.deltaY > 0 ? 1 : e.deltaY < 0 ? -1 : 0;
    if (!dir) return;
    const step = e.shiftKey ? 10 : 1;
    setFrame(state.frame + dir * step, { scrub: true });
  }, { passive: false });
}

function bindControls() {
  els.btnPrev.addEventListener("click", () => stepFrame(-1));
  els.btnNext.addEventListener("click", () => stepFrame(1));
  els.btnPrev10.addEventListener("click", () => stepFrame(-10));
  els.btnNext10.addEventListener("click", () => stepFrame(10));
  els.btnPlay.addEventListener("click", togglePlay);

  els.timeline.addEventListener("input", (e) => setFrame(+e.target.value, { scrub: true }));
  els.frameInput.addEventListener("change", (e) => setFrame(+e.target.value));
  els.episodeSelect.addEventListener("change", (e) => {
    loadEpisode(+e.target.value).catch(err => showToast(err.message, true));
  });

  els.btnEditLabels.addEventListener("click", () => toggleLabelEditor(!state.editingLabels));
  els.btnCloseEdit.addEventListener("click", () => toggleLabelEditor(false));
  els.labelAddForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const v = els.labelAddInput.value.trim();
    if (!v) return;
    if (state.labels.includes(v)) { showToast("label already exists", true); return; }
    await saveLabels([...state.labels, v]);
    els.labelAddInput.value = "";
    els.labelAddInput.focus();
  });

  const savedAnnotator = localStorage.getItem("annotator") || "";
  els.annotator.value = savedAnnotator;
  els.annotator.addEventListener("change", () => {
    localStorage.setItem("annotator", els.annotator.value.trim());
  });
}

async function init() {
  try {
    state.info = await fetchJSON("/api/info");
    const [episodes, labelsResp] = await Promise.all([
      fetchJSON("/api/episodes"),
      fetchJSON("/api/labels"),
    ]);
    state.labels = labelsResp.labels || [];
    for (const e of episodes) {
      const opt = document.createElement("option");
      opt.value = String(e.episode_index);
      const badge = e.subtask_count > 0 ? ` [${e.subtask_count}]` : "";
      opt.textContent = `ep ${String(e.episode_index).padStart(3, "0")} \u00B7 ${e.length}f${badge}`;
      els.episodeSelect.appendChild(opt);
    }
    if (episodes.length === 0) { showToast("no episodes", true); return; }
    bindControls();
    bindKeyboard();
    bindScroll();
    await loadEpisode(episodes[0].episode_index);
  } catch (e) {
    showToast(`init failed: ${e.message}`, true);
  }
}

init();
