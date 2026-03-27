import { chart } from '/mlops-charts.js';

chart('keygen_live', (probePaths, ctx, listener) => {
  const canvas = document.createElement('canvas');
  ctx.containerElement.innerHTML = '';
  ctx.containerElement.appendChild(canvas);

  const colors = {
    rsa4096: 'rgb(255, 99, 132)',
    rsa2048: 'rgb(54, 162, 235)',
    ed25519: 'rgb(75, 192, 192)'
  };

  const desiredOrder = [
    'rsa_4096_bench',
    'rsa_2048_bench',
    'ed25519_bench'
  ];

  const keys = Object.keys(probePaths || {});
  const selectedKeys = desiredOrder
    .map((name) => keys.find((k) => (k.split('/')[0] || k) === name))
    .filter(Boolean);

  const prettyLabel = (canonicalPath, fallback) => {
    const raw = String(canonicalPath || '');
    const procMatch = raw.match(/process\[@name='([^']+)'\]/);
    if (procMatch && procMatch[1]) return procMatch[1];
    return String(fallback || raw || 'process');
  };

  const pickColor = (label) => {
    const l = String(label || '').toLowerCase();
    if (l.includes('rsa_4096')) return colors.rsa4096;
    if (l.includes('rsa_2048')) return colors.rsa2048;
    if (l.includes('ed25519')) return colors.ed25519;
    return 'rgb(153, 102, 255)';
  };

  const chartData = {
    labels: Array.from({ length: 101 }, (_, i) => i),
    datasets: selectedKeys.map((k) => {
      const canonical = probePaths[k];
      const baseLabel = prettyLabel(canonical, k).replace(/_/g, ' ');
      const c = pickColor(baseLabel);
      return {
        label: `${baseLabel.toUpperCase()} (n=0)`,
        _baseLabel: baseLabel.toUpperCase(),
        data: [],
        _bandData: [],
        _targetTrials: null,
        borderColor: c,
        backgroundColor: c + '33',
        tension: 0.15,
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 2,
        spanGaps: true
      };
    })
  };

  const normalizeSeriesToPercent = (series, targetTrials, points = 101) => {
    const values = Array.isArray(series) ? series.filter((v) => Number.isFinite(v)) : [];
    const n = values.length;
    if (n === 0) return new Array(points).fill(null);
    const total = targetTrials && targetTrials > 0 ? targetTrials : n;
    const out = new Array(points).fill(null);
    for (let i = 0; i < n; i += 1) {
      const pct = (i / (total - 1)) * (points - 1);
      const bucket = Math.round(pct);
      if (bucket < points) out[bucket] = values[i];
    }
    // forward-fill gaps between mapped points
    for (let i = 1; i < points; i += 1) {
      if (out[i] === null && out[i - 1] !== null) out[i] = out[i - 1];
    }
    return out;
  };

  const percentile = (arr, p) => {
    if (!arr.length) return null;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.min(sorted.length - 1, lo + 1);
    const w = idx - lo;
    return sorted[lo] * (1 - w) + sorted[hi] * w;
  };

  const rollingStats = (values, windowSize = 10) => {
    const n = values.length;
    const med = new Array(n).fill(null);
    const p10 = new Array(n).fill(null);
    const p90 = new Array(n).fill(null);
    for (let i = 0; i < n; i += 1) {
      const start = Math.max(0, i - windowSize + 1);
      const slice = values.slice(start, i + 1).filter((v) => Number.isFinite(v));
      if (!slice.length) continue;
      med[i] = percentile(slice, 0.5);
      p10[i] = percentile(slice, 0.1);
      p90[i] = percentile(slice, 0.9);
    }
    return { med, p10, p90 };
  };

  const allDatasetsWithBands = chartData.datasets.flatMap((ds, idx) => {
    const lowerBand = {
      label: `${ds._baseLabel} P10`,
      data: [],
      borderColor: 'rgba(0,0,0,0)',
      backgroundColor: ds.backgroundColor,
      fill: idx * 3 + 1,
      pointRadius: 0,
      tension: 0.15,
      hidden: false
    };
    const upperBand = {
      label: `${ds._baseLabel} P90`,
      data: [],
      borderColor: 'rgba(0,0,0,0)',
      backgroundColor: ds.backgroundColor,
      fill: false,
      pointRadius: 0,
      tension: 0.15,
      hidden: false
    };
    ds.fill = false;
    return [upperBand, lowerBand, ds];
  });

  const chartInstance = new Chart(canvas, {
    type: 'line',
    data: { labels: chartData.labels, datasets: allDatasetsWithBands },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Progress (%)' } },
        y: { title: { display: true, text: 'Key generation (ms)' }, beginAtZero: true }
      },
      plugins: {
        title: { display: true, text: 'Asymmetric Key Generation Latency (Live, Progress-Normalized)' },
        legend: {
          display: true,
          labels: {
            filter: (item, data) => {
              const ds = data.datasets[item.datasetIndex];
              return !String(ds.label || '').includes(' P10') && !String(ds.label || '').includes(' P90');
            }
          }
        }
      },
      animation: false
    }
  });
  ctx.setChartInstance(chartInstance);

  listener.subscribeAll(probePaths, (allMetrics) => {
    chartData.datasets.forEach((dataset, idx) => {
      const probeKey = selectedKeys[idx];
      const m = (allMetrics || {})[probeKey] || {};
      const series = ctx.toSeries((m.keygen_ms || {}));
      const targetTrials = Number(m.num_trials || m.target_trials || 0) || null;
      const normalized = normalizeSeriesToPercent(series, targetTrials, 101);
      const stats = rollingStats(normalized, 10);
      dataset.data = stats.med;
      dataset._bandData = { lower: stats.p10, upper: stats.p90 };
      dataset._targetTrials = targetTrials;
      dataset.label = `${dataset._baseLabel} (n=${series.length}${targetTrials ? `/${targetTrials}` : ''})`;
    });

    chartInstance.data.datasets.forEach((ds, dsIdx) => {
      const mapped = Math.floor(dsIdx / 3);
      const role = dsIdx % 3; // 0: upper, 1: lower, 2: median
      const main = chartData.datasets[mapped];
      if (!main) return;
      if (role === 0) ds.data = main._bandData?.upper || [];
      if (role === 1) ds.data = main._bandData?.lower || [];
      if (role === 2) ds.data = main.data || [];
      if (role === 2) ds.label = main.label;
    });

    chartInstance.update('none');
  });
});

