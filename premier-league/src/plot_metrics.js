import { chart } from '/mlops-charts.js';

// Dynamic chart: NN A vs NN B vs NN C losses on same chart
chart('nn_losses', (probePaths, ctx, listener) => {
  const canvas = document.createElement('canvas');
  ctx.containerElement.innerHTML = '';
  ctx.containerElement.appendChild(canvas);

  const colors = [
    'rgb(75, 192, 192)',
    'rgb(255, 99, 132)',
    'rgb(54, 162, 235)'
  ];

  const chartData = {
    labels: [],
    datasets: []
  };

  const formatProbeLabel = (canonicalPath) => {
    const raw = String(canonicalPath || '');
    const partMatch = raw.match(/@partition='p(\d+)'/);
    const seedMatch = raw.match(/@seed='(\d+)'/);
    const procMatch = raw.match(/process\[@name='([^']+)'\]/);
    const parts = [];
    if (procMatch && procMatch[1]) {
      parts.push(procMatch[1]);
    }
    if (partMatch && partMatch[1]) {
      parts.push(`P${partMatch[1]}`);
    }
    if (seedMatch && seedMatch[1]) {
      parts.push(`S${seedMatch[1]}`);
    }
    return parts.join(' ') || raw;
  };

  const desiredOrder = [
    'nn_a_p1_seed41',
    'nn_a_p2_seed41',
    'nn_a_p1_seed42'
  ];

  let colorIndex = 0;
  const keys = Object.keys(probePaths);
  const selectedKeys = desiredOrder
    .map((procName) => keys.find((k) => (k.split('/')[0] || k) === procName))
    .filter(Boolean);

  selectedKeys.forEach((k) => {
    const canonical = probePaths[k];
    chartData.datasets.push({
      label: formatProbeLabel(canonical).replace(/_/g, ' ').toUpperCase(),
      data: [],
      borderColor: colors[colorIndex % colors.length],
      backgroundColor: colors[colorIndex % colors.length] + '33',
      tension: 0.1,
      fill: false
    });
    colorIndex++;
  });

  const chartInstance = new Chart(canvas, {
    type: 'line',
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Epoch' } },
        y: { title: { display: true, text: 'Loss' }, beginAtZero: false }
      },
      plugins: {
        title: { display: true, text: 'NN Training Loss (A)' },
        legend: { display: true }
      },
      animation: false
    }
  });
  ctx.setChartInstance(chartInstance);

  listener.subscribeAll(probePaths, (allMetrics) => {
    let maxLength = 0;

    chartData.datasets.forEach((dataset, idx) => {
      const probeKey = selectedKeys[idx];
      const metrics = allMetrics[probeKey] || {};
      const lossSeries = ctx.toSeries(metrics.train_loss || {});
      dataset.data = lossSeries;
      maxLength = Math.max(maxLength, lossSeries.length);
    });

    chartData.labels = Array.from({ length: maxLength }, (_, i) => i + 1);

    chartInstance.update('none');
  });
});


