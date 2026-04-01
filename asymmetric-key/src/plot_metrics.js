import { chart } from '/mlops-charts.js';

chart('rsa_p2_timing_comparison', (probePaths, ctx, listener) => {
  const canvas = document.createElement('canvas');
  ctx.containerElement.innerHTML = '';
  ctx.containerElement.appendChild(canvas);

  const chartData = {
    labels: [],
    datasets: [
      {
        label: 'RSA 1024 Encrypt',
        data: [],
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.2,
        fill: false
      },
      {
        label: 'RSA 1024 Decrypt',
        data: [],
        borderColor: 'rgb(54, 162, 235)',
        borderDash: [6, 4],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.2,
        fill: false
      },
      {
        label: 'RSA 2048 Encrypt',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.2,
        fill: false
      },
      {
        label: 'RSA 2048 Decrypt',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        borderDash: [6, 4],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.2,
        fill: false
      }
    ]
  };

  const chartInstance = new Chart(canvas, {
    type: 'line',
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Trial' }
        },
        y: {
          title: { display: true, text: 'Time (ms)' },
          beginAtZero: true
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'RSA p2 Trial Timing Comparison (1024 vs 2048)'
        },
        legend: { display: true }
      },
      animation: false
    }
  });
  ctx.setChartInstance(chartInstance);

  listener.subscribeAll(probePaths, (allMetrics) => {
    const m1024 = allMetrics.rsa_1024_p2 || {};
    const m2048 = allMetrics.rsa_2048_p2 || {};

    const enc1024 = ctx.toSeries(m1024.encrypt_ms || {});
    const dec1024 = ctx.toSeries(m1024.decrypt_ms || {});
    const enc2048 = ctx.toSeries(m2048.encrypt_ms || {});
    const dec2048 = ctx.toSeries(m2048.decrypt_ms || {});

    chartData.datasets[0].data = enc1024;
    chartData.datasets[1].data = dec1024;
    chartData.datasets[2].data = enc2048;
    chartData.datasets[3].data = dec2048;

    const maxLength = Math.max(enc1024.length, dec1024.length, enc2048.length, dec2048.length);
    chartData.labels = Array.from({ length: maxLength }, (_, i) => i + 1);

    chartInstance.update('none');
  });
});
