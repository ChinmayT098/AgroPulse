/**
 * AgroPulse - In-browser Random Forest inference
 * Used by the static Netlify build, where no Flask server is available.
 * Runs the same trained forest exported by build_static.py and mirrors
 * CropYieldModelManager.predict() from model.py.
 */

(function () {
  const form = document.getElementById('predictForm');
  if (!form || !form.dataset.staticModel) return;

  let modelPromise = null;
  function loadModel() {
    if (!modelPromise) {
      modelPromise = fetch(form.dataset.staticModel).then(res => {
        if (!res.ok) throw new Error('Unable to load the trained model.');
        return res.json();
      });
    }
    return modelPromise;
  }
  loadModel().catch(() => {});

  // One-hot encode categoricals (unknown categories -> all zeros), then numericals
  function buildFeatureVector(model, input) {
    const x = [];
    model.categorical_features.forEach(col => {
      model.categories[col].forEach(cat => x.push(input[col] === cat ? 1 : 0));
    });
    model.numerical_features.forEach(col => x.push(input[col]));
    // sklearn trees compare float32 inputs
    return x.map(v => Math.fround(v));
  }

  function predictTree(tree, x) {
    let node = 0;
    while (tree.l[node] !== -1) {
      node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  }

  function round(value, digits) {
    const factor = Math.pow(10, digits);
    return Math.round(value * factor) / factor;
  }

  function predict(model, input) {
    const x = buildFeatureVector(model, input);
    const total = model.trees.reduce((sum, tree) => sum + predictTree(tree, x), 0);
    const predYield = Math.max(0.05, round(total / model.trees.length, 2));

    const area = input.Area_Hectares;
    const stats = model.crop_stats[input.Crop] || { avg_yield: predYield };
    const avgYield = stats.avg_yield;
    const diffPct = round(((predYield - avgYield) / (avgYield + 1e-6)) * 100, 1);

    let status = 'Close to Historical Average';
    let statusClass = 'text-primary';
    if (diffPct > 7.0) {
      status = 'Above Historical Average';
      statusClass = 'text-success';
    } else if (diffPct < -7.0) {
      status = 'Below Historical Average';
      statusClass = 'text-warning';
    }

    return {
      crop: input.Crop,
      predicted_yield: predYield,
      area_hectares: area,
      estimated_production: round(predYield * area, 2),
      historical_avg_yield: avgYield,
      comparison_percent: diffPct,
      comparison_status: status,
      status_class: statusClass
    };
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function renderResult(p) {
    const sign = p.comparison_percent >= 0 ? '+' : '';
    const fmt = (n, digits) => n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    document.getElementById('clientPrediction').innerHTML = `
      <div class="prediction-result-banner" id="predictionResultCard">
        <div class="result-header">
          <div>
            <span class="badge badge-green" style="margin-bottom: 6px;">Prediction Generated</span>
            <h3 style="font-size: 20px; color: #065f46; font-weight: 700;">Forecasted Agricultural Yield for ${escapeHtml(p.crop)}</h3>
          </div>
          <div class="result-yield-display">
            <div class="result-yield-number">${p.predicted_yield}</div>
            <div class="result-yield-unit">Tons / Hectare</div>
          </div>
        </div>
        <div class="result-details-grid">
          <div class="result-mini-card">
            <div class="result-mini-label">Cultivation Area</div>
            <div class="result-mini-val">${p.area_hectares.toLocaleString('en-US')} ha</div>
          </div>
          <div class="result-mini-card">
            <div class="result-mini-label">Estimated Total Production</div>
            <div class="result-mini-val" style="color: #059669;">${fmt(p.estimated_production, 1)} Tons</div>
          </div>
          <div class="result-mini-card">
            <div class="result-mini-label">Historical Crop Average</div>
            <div class="result-mini-val">${p.historical_avg_yield} t/ha</div>
          </div>
          <div class="result-mini-card">
            <div class="result-mini-label">Historical Comparison</div>
            <div class="result-mini-val" style="font-size: 14px;">
              <span class="${p.status_class}">${escapeHtml(p.comparison_status)}</span>
              (${sign}${p.comparison_percent.toFixed(1)}%)
            </div>
          </div>
        </div>
      </div>
      <div style="margin-bottom: 24px;"></div>`;
    document.getElementById('predictionResultCard').scrollIntoView({ behavior: 'smooth' });
  }

  function showError(message) {
    const box = document.getElementById('clientPredictionError');
    box.innerHTML = `<strong>⚠️ Input Validation Error:</strong> ${escapeHtml(message)}`;
    box.style.display = message ? 'block' : 'none';
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    showError('');

    try {
      const model = await loadModel();
      const data = new FormData(form);
      const input = {};
      model.categorical_features.forEach(col => { input[col] = data.get(col); });
      model.numerical_features.forEach(col => {
        const val = parseFloat(data.get(col));
        if (Number.isNaN(val)) throw new Error(`Invalid numerical value provided for feature '${col}'.`);
        if (val < 0) throw new Error(`Value for ${col.replace(/_/g, ' ')} cannot be negative.`);
        input[col] = val;
      });
      renderResult(predict(model, input));
    } catch (err) {
      showError(err.message);
      document.getElementById('clientPredictionError').scrollIntoView({ behavior: 'smooth' });
    }
  });

  form.addEventListener('reset', () => {
    showError('');
    document.getElementById('clientPrediction').innerHTML = '';
  });
})();
