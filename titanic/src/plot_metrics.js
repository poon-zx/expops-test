import { chart } from '/mlops-charts.js';

chart('nn_losses', (probePaths, ctx, listener) => {
  const canvas = document.createElement('canvas');
  ctx.containerElement.innerHTML = '';
  ctx.containerElement.appendChild(canvas);

  const desiredOrder = ['nn_training_a', 'nn_training_b'];
  const keys = Object.keys(probePaths || {});
  const selectedKeys = desiredOrder
    .map((name) => keys.find((k) => (k.split('/')[0] || k) === name))
    .filter(Boolean);

  const colors = [
    'rgb(54, 162, 235)',
    'rgb(255, 99, 132)',
  ];

  const prettyLabel = (canonicalPath, fallback) => {
    const raw = String(canonicalPath || '');
    const procMatch = raw.match(/process\[@name='([^']+)'\]/);
    if (procMatch && procMatch[1]) return procMatch[1];
    return String(fallback || raw || 'process');
  };

  const chartData = {
    labels: [],
    datasets: selectedKeys.map((k, idx) => {
      const canonical = probePaths[k];
      const label = prettyLabel(canonical, k).replace(/_/g, ' ');
      const c = colors[idx % colors.length];
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
        x: { title: { display: true, text: 'Epoch' } },
        y: { title: { display: true, text: 'Train loss' }, beginAtZero: false }
      },
      plugins: {
        title: { display: true, text: 'NN Training Loss (Titanic)' },
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
      const series = ctx.toSeries((m.train_loss || {}));
      dataset.data = series;
      maxLen = Math.max(maxLen, series.length);
    });

    chartData.labels = Array.from({ length: maxLen }, (_, i) => i + 1);
    chartInstance.update('none');
  });
});

