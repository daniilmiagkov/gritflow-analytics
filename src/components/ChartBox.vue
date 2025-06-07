<template>
  <div class="chart-box">
    <h4>{{ title }}</h4>
    <canvas ref="canvas" width="300" height="200"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, defineProps } from 'vue';
import { Chart } from 'chart.js';

const props = defineProps<{ title: string; values: number[] }>();
const canvas = ref<HTMLCanvasElement|null>(null);
let chartInstance: Chart|null = null;

function updateHistogram(vals: number[]) {
  if (!chartInstance) return;
  if (!vals.length) {
    chartInstance.data.labels = [];
    chartInstance.data.datasets[0].data = [];
  } else {
    // простая гистограмма «по точкам»
    const m = new Map<string, number>();
    vals.forEach(v => {
      const k = v.toFixed(1);
      m.set(k, (m.get(k)||0)+1);
    });
    chartInstance.data.labels = Array.from(m.keys());
    chartInstance.data.datasets[0].data = Array.from(m.values());
  }
  chartInstance.update();
}

onMounted(() => {
  if (canvas.value) {
    chartInstance = new Chart(canvas.value.getContext('2d')!, {
      type: 'bar',
      data: { labels: [], datasets: [{ label: 'Частота', data: [], backgroundColor: 'rgba(54,162,235,0.6)', borderColor: 'rgba(54,162,235,1)', borderWidth: 1 }] },
      options: { responsive: false,
        scales: {
          x: { title: { display: true, text: 'Диагональ (мм)' } },
          y: { title: { display: true, text: 'Частота' } }
        }
      }
    });
    updateHistogram(props.values);
    watch(() => props.values, v => updateHistogram(v));
  }
});
</script>

<style scoped>
.chart-box {
  display: flex; flex-direction: column; align-items: center;
}
.chart-box h4 {
  margin: 0 0 5px 0; font-size: 14px;
}
.chart-box canvas {
  background: #f4f4f4; border: 1px solid #ccc;
}
</style>
