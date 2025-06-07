<template>
  <div class="config-checkbox">
    <input type="checkbox" :id="id" :checked="modelValue" @change="$emit('update:modelValue', $event.target.checked)" />
    <label :for="id">{{ label }}</label>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';
const props = defineProps<{
  id: string;
  label: string;
  modelValue: boolean;
}>();
</script>

<style scoped>
.config-checkbox { display: flex; align-items: center; gap: 4px; }
</style>
