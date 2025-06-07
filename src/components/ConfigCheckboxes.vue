<template>
  <div class="checkbox-container">
    <div v-if="depthLabels.length" class="checkbox-group">
      <h4>Depth-конфиги</h4>
      <div v-for="lbl in depthLabels" :key="lbl">
        <input
          type="checkbox"
          :id="`chk-depth_${lbl}`"
          :checked="selectedChecks[`depth_${lbl}`]"
          @change="onChange(`depth_${lbl}`, $event.target.checked)"
        />
        <label :for="`chk-depth_${lbl}`">{{ formatLabel(lbl) }}</label>
      </div>
    </div>
    <div v-if="colorLabels.length" class="checkbox-group">
      <h4>Color-конфиги</h4>
      <div v-for="lbl in colorLabels" :key="lbl">
        <input
          type="checkbox"
          :id="`chk-color_${lbl}`"
          :checked="selectedChecks[`color_${lbl}`]"
          @change="onChange(`color_${lbl}`, $event.target.checked)"
        />
        <label :for="`chk-color_${lbl}`">{{ formatLabel(lbl) }}</label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { defineProps, defineEmits } from 'vue';

const props = defineProps<{
  depthLabels: string[];
  colorLabels: string[];
  selectedChecks: Record<string, boolean>;
}>();

const emit = defineEmits<{
  (e: 'update:selected-checks', payload: { key: string; value: boolean }): void;
}>();

function onChange(key: string, val: boolean) {
  emit('update:selected-checks', { key, value: val });
}

function formatLabel(lbl: string) {
  return lbl
    .split(/[-_]/g)
    .map(w => w[0].toUpperCase() + w.slice(1))
    .join(' ');
}
</script>

<style scoped>
.checkbox-container {
  margin-top: 20px;
  display: flex; flex-wrap: wrap; gap: 20px;
}
.checkbox-group {
  display: flex; flex-direction: column; padding: 8px;
  border: 1px solid #ccc; border-radius: 4px; background: #fafafa;
}
.checkbox-group h4 {
  margin: 0 0 5px 0; font-size: 14px;
}
</style>
