/**
 * batch.js — Batch file processing for X-ray analysis
 */

const dropZone = document.getElementById('batch-drop-zone');
const fileInput = document.getElementById('batch-file-input');
const fileList = document.getElementById('file-list');
const fileCount = document.getElementById('file-count');
const analyseBatchBtn = document.getElementById('analyse-batch-btn');
const clearBatchBtn = document.getElementById('clear-batch-btn');
const batchProgress = document.getElementById('batch-progress');
const resultsPreview = document.getElementById('results-preview');
const progressText = document.getElementById('progress-text');
const progressFill = document.getElementById('progress-fill');

let selectedFiles = [];

// File selection handlers
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

function handleFileSelect(e) {
  handleFiles(e.target.files);
}

function handleFiles(files) {
  selectedFiles = Array.from(files).filter(f => {
    const valid = f.size <= 16 * 1024 * 1024 && ['image/jpeg', 'image/png'].includes(f.type);
    if (!valid) {
      alert(`${f.name}: Invalid file. JPG/PNG only, max 16 MB.`);
    }
    return valid;
  });

  updateFileList();
}

function updateFileList() {
  fileCount.textContent = `${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected`;
  fileList.innerHTML = '';

  selectedFiles.forEach((file, idx) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
      <div class="file-info">
        <p class="file-name">${file.name}</p>
        <p class="file-size">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
      </div>
      <button type="button" class="btn-remove" onclick="removeFile(${idx})">✕</button>
    `;
    fileList.appendChild(item);
  });

  analyseBatchBtn.disabled = selectedFiles.length === 0;
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  updateFileList();
}

clearBatchBtn.addEventListener('click', () => {
  selectedFiles = [];
  fileInput.value = '';
  fileList.innerHTML = '';
  updateFileList();
});

analyseBatchBtn.addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;

  // Show progress
  dropZone.classList.add('hidden');
  batchProgress.classList.remove('hidden');

  const formData = new FormData();
  selectedFiles.forEach(file => {
    formData.append('files', file);
  });

  try {
    const response = await fetch('/api/predict-batch', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.status !== 'success') {
      showError(data.error || 'Batch processing failed');
      return;
    }

    // Display results
    displayBatchResults(data.results, data.errors);

    // Reset
    selectedFiles = [];
    fileInput.value = '';
  } catch (error) {
    showError('Network error: ' + error.message);
  }
});

function displayBatchResults(results, errors) {
  progressText.textContent = `Completed: ${results.length}/${selectedFiles.length + results.length}`;
  progressFill.style.width = `${(results.length / (selectedFiles.length + results.length)) * 100}%`;

  resultsPreview.innerHTML = '';

  results.forEach(result => {
    const item = document.createElement('div');
    item.className = `result-item result-item--${result.label.toLowerCase()}`;
    item.innerHTML = `
      <div class="result-badge">${result.label}</div>
      <div class="result-details">
        <p class="result-file">${result.filename}</p>
        <p class="result-score">Score: ${result.score.toFixed(4)} | Confidence: ${result.confidence}%</p>
      </div>
    `;
    resultsPreview.appendChild(item);
  });

  if (errors.length > 0) {
    const errorTitle = document.createElement('p');
    errorTitle.className = 'errors-title';
    errorTitle.textContent = `${errors.length} file(s) failed:`;
    resultsPreview.appendChild(errorTitle);

    errors.forEach(err => {
      const item = document.createElement('div');
      item.className = 'error-item';
      item.innerHTML = `
        <p class="error-file">${err.file}</p>
        <p class="error-message">${err.error}</p>
      `;
      resultsPreview.appendChild(item);
    });
  }
}

function showError(message) {
  alert('Error: ' + message);
  batchProgress.classList.add('hidden');
  dropZone.classList.remove('hidden');
}
