<template>
  <div id="container">
    <!-- Управление -->
    <Controls
      :start-frame="startFrame"
      :end-frame="endFrame"
      :is-running="!!ws"
      @update-range="(r) => { startFrame = r.start; endFrame = r.end }"
      @start-ws="startWebSocket"
    />

    <!-- Статус и номер кадра -->
    <StatusInfo
      :ws-status="wsStatus"
      :current-frame="currentFrame"
    />

    <!-- Оверлей -->
    <Overlay :overlay-src="overlaySrc" />

    <!-- Чекбоксы, когда первый пакет данных -->
    <ConfigCheckboxes
      v-if="firstMessageHandled"
      :depth-labels="depthLabels"
      :color-labels="colorLabels"
      v-model:selected-checks="selectedChecks"
    />

    <!-- Графики -->
    <ChartsWrapper
      :chart-data="chartData"
      :selected-checks="selectedChecks"
    />

    <!-- Порог -->
    <ThresholdInput
      v-model:threshold="largeThreshold"
      :alert-large="alertLarge"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Controls from './components/Controls.vue';
import StatusInfo from './components/StatusInfo.vue';
import Overlay from './components/Overlay.vue';
import ConfigCheckboxes from './components/ConfigCheckboxes.vue';
import ChartsWrapper from './components/ChartsWrapper.vue';
import ThresholdInput from './components/ThresholdInput.vue';

let ws: WebSocket | null = null;

const wsStatus = ref<'не подключено'|'подключение...'|'подключено'|'отключено'|'ошибка'>('не подключено');
const startFrame = ref(0);
const endFrame   = ref(19);
const currentFrame = ref<number|null>(null);
const overlaySrc = ref<string>('');

const firstMessageHandled = ref(false);
const depthLabels: string[] = [];
const colorLabels: string[] = [];
const selectedChecks = ref<Record<string, boolean>>({});

const chartData = ref<Record<string, any>>({});
const largeThreshold = ref(100);
const alertLarge = ref('');

function startWebSocket() {
  if (ws) return;
  if (startFrame.value < 0 || endFrame.value < startFrame.value) {
    alert('Введите корректный диапазон: end ≥ start, оба ≥ 0.');
    return;
  }
  wsStatus.value = 'подключение...';
  ws = new WebSocket(`ws://${window.location.host}/ws?start=${startFrame.value}&end=${endFrame.value}`);
  ws.onopen = () => { wsStatus.value = 'подключено'; };
  ws.onerror = () => { wsStatus.value = 'ошибка'; };
  ws.onclose = () => { wsStatus.value = 'отключено'; ws = null; };
  ws.onmessage = e => {
    const data = JSON.parse(e.data);
    currentFrame.value = data.frame;
    overlaySrc.value = 'data:image/png;base64,' + data.overlay_b64;

    if (!firstMessageHandled.value) {
      const dSet = new Set<string>();
      const cSet = new Set<string>();
      Object.keys(data).forEach(k => {
        if (k.startsWith('diagonals_depth_') && k.endsWith('_mm'))
          dSet.add(k.slice(16, -3));
        if (k.startsWith('diagonals_color_') && k.endsWith('_mm'))
          cSet.add(k.slice(16, -3));
      });
      depthLabels.push(...dSet);
      colorLabels.push(...cSet);
      depthLabels.forEach(l => selectedChecks.value[`depth_${l}`] = true);
      colorLabels.forEach(l => selectedChecks.value[`color_${l}`] = true);
      firstMessageHandled.value = true;
    }

    // alertLarge
    const thr = largeThreshold.value;
    let found = false;
    Object.entries(data).forEach(([k,v]) => {
      if (Array.isArray(v) && (k.startsWith('diagonals_depth_')||k.startsWith('diagonals_color_')))
        if ((v as number[]).some(x => x >= thr)) found = true;
    });
    alertLarge.value = found ? 'Найдены крупные элементы!' : '';

    chartData.value = data;
  };
}
</script>

<style>
#container {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: Arial, sans-serif;
  margin: 20px;
}
</style>
