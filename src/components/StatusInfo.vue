<template>
  <div>
    <div class="status">Статус: {{ wsStatus }}</div>
    <div class="frame-info">Кадр: {{ currentFrame ?? '—' }}</div>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';

const props = defineProps<{
  wsStatus: string;
  currentFrame: number|null;
}>();

const wsStatus = props.wsStatus;
const currentFrame = props.currentFrame;
</script>

<style scoped>
.status, .frame-info {
  margin-top: 10px;
  font-weight: bold;
}
</style>
