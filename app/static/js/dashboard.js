/**
 * dashboard.js — Analytics dashboard with charts and statistics
 */

let distributionChart, scoreHistogram;

async function loadDashboardData() {
  try {
    // Fetch statistics
    const statsResponse = await fetch('/api/dashboard/stats');
    const stats = await statsResponse.json();

    // Update stat cards
    updateStatsCards(stats);

    // Fetch analyses
    const analysesResponse = await fetch('/api/dashboard/analyses?limit=20');
    const analysesData = await analysesResponse.json();

    // Initialize charts
    initializeCharts(stats, analysesData.analyses);

    // Update recent analyses
    updateRecentAnalyses(analysesData.analyses);
  } catch (error) {
    console.error('Error loading dashboard:', error);
  }
}

function updateStatsCards(stats) {
  document.getElementById('total-analyses').textContent = stats.total;
  document.getElementById('normal-count').textContent = stats.normal;
  document.getElementById('anomaly-count').textContent = stats.anomalies;

  document.getElementById('stat-total').textContent = stats.total;
  document.getElementById('stat-normal').textContent = stats.normal;
  document.getElementById('stat-normal-pct').textContent = `${stats.normal_percentage}%`;

  document.getElementById('stat-anomaly').textContent = stats.anomalies;
  document.getElementById('stat-anomaly-pct').textContent = `${stats.anomaly_percentage}%`;

  document.getElementById('stat-confidence').textContent = `${stats.avg_confidence}%`;
}

function initializeCharts(stats, analyses) {
  // Pie chart for classification distribution
  const pieCtx = document.getElementById('distribution-chart').getContext('2d');
  const pieData = {
    labels: ['Normal', 'Anomalies'],
    datasets: [{
      data: [stats.normal, stats.anomalies],
      backgroundColor: [
        'rgba(15, 125, 61, 0.8)',  // Teal
        'rgba(163, 45, 45, 0.8)',  // Red
      ],
      borderColor: [
        'rgba(15, 125, 61, 1)',
        'rgba(163, 45, 45, 1)',
      ],
      borderWidth: 2,
    }],
  };

  distributionChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: pieData,
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: 'bottom' },
      },
    },
  });

  // Histogram of anomaly scores
  const scores = analyses.map(a => (a.score * 100).toFixed(0));
  const bins = createHistogramBins(scores, 10);

  const histCtx = document.getElementById('score-histogram').getContext('2d');
  const histData = {
    labels: bins.map((_, i) => `${i * 10}-${(i + 1) * 10}`),
    datasets: [{
      label: 'Frequency',
      data: bins,
      backgroundColor: 'rgba(24, 95, 165, 0.8)',
      borderColor: 'rgba(24, 95, 165, 1)',
      borderWidth: 1,
    }],
  };

  scoreHistogram = new Chart(histCtx, {
    type: 'bar',
    data: histData,
    options: {
      responsive: true,
      maintainAspectRatio: true,
      indexAxis: 'x',
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

function createHistogramBins(scores, numBins) {
  const bins = Array(numBins).fill(0);
  scores.forEach(score => {
    const binIdx = Math.min(Math.floor(score / 10), numBins - 1);
    bins[binIdx]++;
  });
  return bins;
}

function updateRecentAnalyses(analyses) {
  const list = document.getElementById('analyses-list');
  list.innerHTML = '';

  if (analyses.length === 0) {
    list.innerHTML = '<p class="empty-state">No analyses yet.</p>';
    return;
  }

  analyses.slice(0, 10).forEach(analysis => {
    const item = document.createElement('div');
    item.className = `analysis-item analysis-item--${analysis.label.toLowerCase()}`;
    const timestamp = new Date(analysis.timestamp).toLocaleString();
    item.innerHTML = `
      <div class="analysis-badge">${analysis.label}</div>
      <div class="analysis-info">
        <p class="analysis-name">${analysis.filename || 'Unknown file'}</p>
        <p class="analysis-meta">${timestamp} • Score: ${(analysis.score * 100).toFixed(2)}% • Confidence: ${analysis.confidence}%</p>
      </div>
    `;
    list.appendChild(item);
  });
}

document.getElementById('export-csv').addEventListener('click', async () => {
  const response = await fetch('/api/dashboard/analyses?limit=1000');
  const data = await response.json();

  let csv = 'Timestamp,Filename,Label,Score,Confidence\n';
  data.analyses.forEach(a => {
    csv += `${a.timestamp},${a.filename || 'unknown'},${a.label},${a.score},${a.confidence}\n`;
  });

  downloadFile(csv, 'analyses_report.csv', 'text/csv');
});

document.getElementById('export-pdf').addEventListener('click', () => {
  alert('PDF export coming soon! For now, use "Print to PDF" from your browser.');
});

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
}

// Load data on page load
document.addEventListener('DOMContentLoaded', loadDashboardData);

// Refresh every 30 seconds
setInterval(loadDashboardData, 30000);
