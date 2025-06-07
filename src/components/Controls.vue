<template>
  <div class="controls">
    <label>
      Стартовый кадр:
      <input type="number" v-model.number="localStart" min="0" />
    </label>
    <label>
      Конечный кадр:
      <input type="number" v-model.number="localEnd" min="0" />
    </label>
    <button :disabled="isRunning" @click="$emit('start-ws')">
      Запустить трансляцию
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, defineProps, defineEmits } from 'vue';

const props = defineProps<{
  startFrame: number;
  endFrame: number;
  isRunning: boolean;
}>();

const emit = defineEmits<{
  (e: 'update-range', payload: { start: number; end: number }): void;
  (e: 'start-ws'): void;
}>();

const localStart = ref(props.startFrame);
const localEnd   = ref(props.endFrame);

watch(() => localStart.value, v => emit('update-range',{ start: v, end: localEnd.value }));
watch(() => localEnd.value,   v => emit('update-range',{ start: localStart.value, end: v }));
</script>

<style scoped>
.controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}
input[type="number"] {
  width: 80px; padding: 5px; font-size: 14px;
}
button:disabled {
  opacity: 0.6; cursor: not-allowed;
}
</style>
