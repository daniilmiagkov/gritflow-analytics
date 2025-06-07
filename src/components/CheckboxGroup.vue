<template>
  <div class="checkbox-group">
    <h4>{{ title }}</h4>
    <slot />
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';
const props = defineProps<{ title: string }>();
</script>

<style scoped>
.checkbox-group {
  display: flex; flex-direction: column; padding: 8px;
  border: 1px solid #ccc; border-radius: 4px; background: #fafafa;
}
.checkbox-group h4 {
  margin: 0 0 5px 0; font-size: 14px;
}
</style>
