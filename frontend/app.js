const runButton = document.getElementById('runButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const errorMessage = document.getElementById('errorMessage');
const emptyState = document.getElementById('emptyState');
const chart = document.getElementById('predictionChart');

const money = (value) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const shortDate = (value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });

function drawChart(points) {
  const context = chart.getContext('2d');
  const width = chart.clientWidth;
  const height = chart.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  chart.width = width * ratio;
  chart.height = height * ratio;
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  const values = points.flatMap((point) => [point.actual, point.predicted]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const padding = { top: 18, right: 12, bottom: 34, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (index) => padding.left + (index / Math.max(points.length - 1, 1)) * plotWidth;
  const y = (value) => padding.top + (1 - (value - minimum) / Math.max(maximum - minimum, 1)) * plotHeight;

  context.font = '11px DM Mono, monospace';
  context.strokeStyle = '#e6eee9';
  context.fillStyle = '#7a8983';
  context.lineWidth = 1;
  for (let index = 0; index < 4; index += 1) {
    const gridY = padding.top + (index / 3) * plotHeight;
    context.beginPath(); context.moveTo(padding.left, gridY); context.lineTo(width - padding.right, gridY); context.stroke();
    context.fillText(money(maximum - ((maximum - minimum) * index) / 3).replace('.00', ''), 0, gridY + 4);
  }
  points.forEach((point, index) => {
    if (index % Math.max(Math.floor(points.length / 5), 1) === 0) context.fillText(shortDate(point.date), x(index) - 18, height - 8);
  });

  const drawLine = (key, color, dashed = false) => {
    context.beginPath(); context.strokeStyle = color; context.lineWidth = 2; context.setLineDash(dashed ? [5, 5] : []);
    points.forEach((point, index) => index === 0 ? context.moveTo(x(index), y(point[key])) : context.lineTo(x(index), y(point[key])));
    context.stroke(); context.setLineDash([]);
  };
  drawLine('actual', '#16734c'); drawLine('predicted', '#e77b42', true);
}

function renderResult(result) {
  runPrediction.lastResult = result;
  const latest = result.points[result.points.length - 1];
  document.getElementById('mae').textContent = money(result.mae);
  document.getElementById('latestActual').textContent = money(latest.actual);
  document.getElementById('latestPrediction').textContent = money(latest.predicted);
  document.getElementById('observations').textContent = result.points.length.toLocaleString();
  document.getElementById('latestDate').textContent = `As of ${shortDate(latest.date)}`;
  document.getElementById('runDate').textContent = `Loaded ${new Date().toLocaleTimeString()}`;
  emptyState.hidden = true;
  drawChart(result.points);

  document.getElementById('predictionRows').innerHTML = result.points.slice().reverse().map((point) => {
    const variance = point.predicted - point.actual;
    const className = variance >= 0 ? 'variance-positive' : 'variance-negative';
    return `<tr><td>${shortDate(point.date)}</td><td>${money(point.actual)}</td><td>${money(point.predicted)}</td><td class="${className}">${variance >= 0 ? '+' : ''}${money(variance)}</td></tr>`;
  }).join('');
}

async function runPrediction() {
  runButton.disabled = true; errorMessage.textContent = '';
  statusText.textContent = 'Running model...'; statusDot.style.background = '#e77b42';
  try {
    const response = await fetch('/api/predictions', { method: 'POST' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The model run failed.');
    renderResult(result);
    statusText.textContent = 'Model run complete'; statusDot.style.background = '#16734c';
  } catch (error) {
    errorMessage.textContent = error.message;
    statusText.textContent = 'Run failed'; statusDot.style.background = '#b64032';
  } finally { runButton.disabled = false; }
}

runButton.addEventListener('click', runPrediction);
window.addEventListener('resize', () => {
  const rows = document.querySelectorAll('#predictionRows tr');
  if (rows.length > 1) runPrediction.lastResult && drawChart(runPrediction.lastResult.points);
});
