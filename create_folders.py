import os

# Определяем структуру проекта и шаблоны содержимого
project_files = {
    "src/App.vue": """<template>
  <div id="app">
    <!-- Твой корневой компонент -->
  </div>
</template>

<script setup lang="ts">
// Импортируй и регистрируй дочерние компоненты здесь, если нужно
</script>

<style scoped>
/* Стили для App.vue */
</style>
""",

    "src/main.js": """import { createApp } from 'vue';
import App from './App.vue';

const app = createApp(App);
app.mount('#app');
""",

    "src/components/Controls.vue": """<template>
  <div class="controls">
    <!-- Поля startFrame и endFrame, кнопка -->
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const startFrame = ref(0);
const endFrame = ref(0);

function startTransmission() {
  // Логика запуска
}
</script>

<style scoped>
/* Стили для Controls.vue */
</style>
""",

    "src/components/StatusInfo.vue": """<template>
  <div class="status-info">
    <!-- Статус WebSocket и номер кадра -->
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const webSocketStatus = ref('не подключено');
const currentFrameNumber = ref<number|null>(null);
</script>

<style scoped>
/* Стили для StatusInfo.vue */
</style>
""",

    "src/components/Overlay.vue": """<template>
  <div class="overlay">
    <slot />
  </div>
</template>

<script setup lang="ts">
// Этот компонент обёртка для изображения
</script>

<style scoped>
/* Стили для Overlay.vue */
</style>
""",

    "src/components/ConfigCheckboxes.vue": """<template>
  <div class="config-checkboxes">
    <!-- Группы чекбоксов для depth и color -->
  </div>
</template>

<script setup lang="ts">
import { defineProps, defineEmits } from 'vue';

const props = defineProps({
  depthLabels: Array as () => string[],
  colorLabels: Array as () => string[],
  selectedChecks: Object as () => Record<string, boolean>,
});

const emit = defineEmits<{
  (e: 'update:selected-checks', payload: { key: string; value: boolean }): void;
}>();

function formatLabel(safeLabel: string) {
  return safeLabel
    .split(/[-_]/g)
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
</script>

<style scoped>
/* Стили для ConfigCheckboxes.vue */
</style>
""",

    "src/components/CheckboxGroup.vue": """<template>
  <div class="checkbox-group">
    <h4>{{ title }}</h4>
    <slot />
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';

const props = defineProps<{
  title: string;
}>();
</script>

<style scoped>
/* Стили для CheckboxGroup.vue */
</style>
""",

    "src/components/ConfigCheckbox.vue": """<template>
  <div>
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
/* Стили для ConfigCheckbox.vue */
</style>
""",

    "src/components/ChartsWrapper.vue": """<template>
  <div class="charts-wrapper">
    <slot />
  </div>
</template>

<script setup lang="ts">
// Обёртка для компонентов ChartBox
</script>

<style scoped>
/* Стили для ChartsWrapper.vue */
</style>
""",

    "src/components/ChartBox.vue": """<template>
  <div class="chart-box">
    <h4>{{ title }}</h4>
    <canvas ref="canvas" width="300" height="200"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue';
import Chart from 'chart.js/auto';

const props = defineProps<{
  title: string;
  values: number[];
}>();

const canvas = ref<HTMLCanvasElement|null>(null);
let chartInstance: Chart | null = null;

onMounted(() => {
  if (canvas.value) {
    chartInstance = new Chart(canvas.value.getContext('2d')!, {
      type: 'bar',
      data: { labels: [], datasets: [{ label: 'Частота', data: [], backgroundColor: 'rgba(54,162,235,0.6)', borderWidth: 1 }] },
      options: { responsive: false }
    });
    updateHistogram(props.values);
  }
});

watch(() => props.values, (newVals) => {
  updateHistogram(newVals);
});

function updateHistogram(values: number[]) {
  if (!chartInstance) return;
  // Здесь можно вставить логику построения гистограммы
  chartInstance.data.labels = values.map(v => v.toFixed(1));
  chartInstance.data.datasets[0].data = values;
  chartInstance.update();
}
</script>

<style scoped>
/* Стили для ChartBox.vue */
</style>
""",

    "src/components/ThresholdInput.vue": """<template>
  <div class="threshold-input">
    <label>Порог (мм):</label>
    <input type="number" v-model.number="localThreshold" min="0" />
    <div class="alert">{{ alertLarge }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, defineProps, defineEmits, watch } from 'vue';

const props = defineProps<{
  threshold: number;
  alertLarge: string;
}>();

const emit = defineEmits<{
  (e: 'update:threshold', value: number): void;
}>();

const localThreshold = ref(props.threshold);

watch(localThreshold, (val) => {
  emit('update:threshold', val);
});
</script>

<style scoped>
/* Стили для ThresholdInput.vue */
</style>
"""
}

def create_files(base_dir: str, files: dict):
    for rel_path, content in files.items():
        abs_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Created: {rel_path}')

if __name__ == "__main__":
    project_root = os.getcwd()  # Или укажи явно путь
    create_files(project_root, project_files)
