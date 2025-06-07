<template>
  <div class="threshold-container">
    <label for="largeThreshold">Порог крупного объекта (мм):</label>
    <input
      id="largeThreshold"
      type="number"
      v-model.number="localThreshold"
      min="0"
      step="1"
    />
    <div class="alert-large">{{ alertLarge }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, defineProps, defineEmits, watch } from 'vue';

const props = defineProps<{ threshold: number; alertLarge: string }>();
const emit = defineEmits<{ (e: 'update:threshold', v: number): void }>();

const localThreshold = ref(props.threshold);
watch(localThreshold, v => emit('update:threshold', v));
</script>

<style scoped>
.threshold-container { margin-top: 10px; }
.alert-large { margin-top: 10px; color: red; font-weight: bold; }
</style>
