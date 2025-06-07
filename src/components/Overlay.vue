<template>
  <div class="overlay-container">
    <img class="segmented-image" :src="overlaySrc" alt="Segmentation Overlay" />
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';

const props = defineProps<{
  overlaySrc: string;
}>();
</script>

<style scoped>
.overlay-container {
  margin-top: 20px;
  border: 1px solid #000;
  aspect-ratio: 800/300;
}
.segmented-image {
  width: 100%; height: 100%; object-fit: contain;
}
</style>
