<script setup lang="ts">
import { computed } from 'vue';
import ChartBox from './ChartBox.vue';

const props = defineProps<{
  chartData: Record<string, any>;
  selectedChecks: Record<string, boolean>;
}>();

// Фильтруем заранее диагональные ключи
const filteredChartEntries = computed(() =>
  Object.entries(props.chartData).filter(([key]) =>
    key.startsWith('diag-'),
  ),
);
</script>

<template>
  <div class="charts-wrapper">
    <ChartBox
      v-for="[key, value] in filteredChartEntries"
      :key="key"
      :chart-data="value"
      :is-active="selectedChecks[key]"
    />
  </div>
</template>
