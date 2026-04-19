/**
 * app/static/js/upload.js
 * ───────────────────────
 * Handles the upload page only:
 *   - Drag-and-drop onto the drop zone
 *   - File input selection
 *   - Client-side validation (type + size)
 *   - Image preview
 *   - POST to /api/predict
 *   - Navigate to result page on success
 *
 * No result rendering here — that lives in result.js.
 */

"use strict";

(() => {
  const MAX_BYTES        = 16 * 1024 * 1024;
  const ALLOWED_TYPES    = ["image/jpeg", "image/png"];
  const ALLOWED_EXTS     = [".jpg", ".jpeg", ".png"];

  // DOM references
  const dropZone         = document.getElementById("drop-zone");
  const fileInput        = document.getElementById("file-input");
  const uploadCard       = document.getElementById("upload-card");
  const previewCard      = document.getElementById("preview-card");
  const previewImg       = document.getElementById("preview-img");
  const previewMeta      = document.getElementById("preview-meta");
  const clearBtn         = document.getElementById("clear-btn");
  const analyseBtn       = document.getElementById("analyse-btn");
  const loadingCard      = document.getElementById("loading-card");
  const errorBanner      = document.getElementById("error-banner");
  const errorMessage     = document.getElementById("error-message");

  let selectedFile = null;

  // ── Drag-and-drop events ─────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ── File input change ────────────────────────────────────────────
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // ── Clear selection ──────────────────────────────────────────────
  clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    Utils.hide(previewCard);
    Utils.show(uploadCard);
    hideError();
  });

  // ── Analyse button ───────────────────────────────────────────────
  analyseBtn.addEventListener("click", () => {
    if (selectedFile) submitFile(selectedFile);
  });

  // ── Core functions ───────────────────────────────────────────────

  function handleFile(file) {
    hideError();
    const err = validateFile(file);
    if (err) { showError(err); return; }

    selectedFile = file;
    showPreview(file);
  }

  function validateFile(file) {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext) && !ALLOWED_TYPES.includes(file.type)) {
      return `Unsupported file type "${ext}". Please upload a JPG or PNG.`;
    }
    if (file.size > MAX_BYTES) {
      return `File too large (${(file.size / 1e6).toFixed(1)} MB). Maximum is 16 MB.`;
    }
    return null;
  }

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src  = e.target.result;
      previewMeta.textContent =
        `${file.name}  ·  ${(file.size / 1024).toFixed(0)} KB`;
      Utils.hide(uploadCard);
      Utils.show(previewCard);
    };
    reader.readAsDataURL(file);
  }

  async function submitFile(file) {
    Utils.hide(previewCard);
    Utils.hide(errorBanner);
    Utils.show(loadingCard);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const result = await Utils.postFormData("/api/predict", formData);

      // Store result for result.html to read
      const originalDataUrl = previewImg.src;
      Utils.storeResult(result, originalDataUrl);

      // Navigate to result page
      window.location.href = "/result";

    } catch (err) {
      Utils.hide(loadingCard);
      Utils.show(previewCard);
      showError(err.message || "Prediction failed. Please try again.");
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    Utils.show(errorBanner);
  }

  function hideError() {
    Utils.hide(errorBanner);
  }

})();