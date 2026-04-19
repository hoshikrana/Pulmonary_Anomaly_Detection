/**
 * app/static/js/result.js
 * ───────────────────────
 * Reads the stored prediction result from sessionStorage and
 * populates the result page DOM.
 *
 * No API calls here — the call already happened in upload.js.
 * No upload logic here.
 * Pure rendering.
 */

"use strict";

(() => {
  const NORMAL_COLOR  = "#1D9E75";
  const ANOMALY_COLOR = "#D85A30";

  function init() {
    const stored = Utils.loadResult();

    if (!stored) {
      // No result in session — redirect to upload
      window.location.href = "/";
      return;
    }

    const { data, originalDataUrl } = stored;
    render(data, originalDataUrl);
    Utils.clearResult();
  }

  function render(data, originalDataUrl) {
    const isAnomaly = data.label !== "Normal";
    const color     = isAnomaly ? ANOMALY_COLOR : NORMAL_COLOR;

    // ── Verdict badge ──────────────────────────────────────────────
    const badge = document.getElementById("verdict-badge");
    badge.textContent = data.label;
    badge.style.background = color + "22";    // 13% opacity fill
    badge.style.color      = color;
    badge.style.border     = `1.5px solid ${color}`;

    // ── Score bar ──────────────────────────────────────────────────
    const scoreBar   = document.getElementById("score-bar");
    const scoreValue = document.getElementById("score-value");
    const pct        = Math.round(data.score * 100);

    scoreValue.textContent = Utils.formatScore(data.score);
    scoreBar.style.width   = pct + "%";
    scoreBar.style.background = color;

    // Animate bar fill
    scoreBar.style.transition = "width 0.8s ease";

    // ── Meta values ────────────────────────────────────────────────
    document.getElementById("confidence-value").textContent = data.confidence;
    document.getElementById("mse-value").textContent        = Utils.formatMSE(data.raw_mse);

    // ── Images ────────────────────────────────────────────────────
    setImage("original-img",       originalDataUrl,           "Original X-ray");
    setImage("heatmap-img",        toDataUrl(data.heatmap_b64),       "Anomaly heatmap");
    setImage("reconstruction-img", toDataUrl(data.reconstruction_b64), "Model reconstruction");
  }

  function setImage(id, src, alt) {
    const img = document.getElementById(id);
    if (img && src) {
      img.src = src;
      img.alt = alt;
    }
  }

  function toDataUrl(b64) {
    return b64 ? `data:image/png;base64,${b64}` : "";
  }

  // Run on page load
  document.addEventListener("DOMContentLoaded", init);
})();