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
    labels: [],
    datasets: selectedKeys.map((k) => {
      const canonical = probePaths[k];
      const label = prettyLabel(canonical, k).replace(/_/g, ' ');
      const c = pickColor(label);
      return {
        label: label.toUpperCase(),
        data: [],
        borderColor: c,
        backgroundColor: c + '33',
        tension: 0.15,
        fill: false
      };
    })
  };

  const chartInstance = new Chart(canvas, {
    type: 'line',
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Trial' } },
        y: { title: { display: true, text: 'Key generation (ms)' }, beginAtZero: true }
      },
      plugins: {
        title: { display: true, text: 'Asymmetric Key Generation Latency (Live)' },
        legend: { display: true }
      },
      animation: false
    }
  });
  ctx.setChartInstance(chartInstance);

  listener.subscribeAll(probePaths, (allMetrics) => {
    let maxLen = 0;

    chartData.datasets.forEach((dataset, idx) => {
      const probeKey = selectedKeys[idx];
      const m = (allMetrics || {})[probeKey] || {};
      const series = ctx.toSeries((m.keygen_ms || {}));
      dataset.data = series;
      maxLen = Math.max(maxLen, series.length);
    });

    chartData.labels = Array.from({ length: maxLen }, (_, i) => i + 1);
    chartInstance.update('none');
  });
});

