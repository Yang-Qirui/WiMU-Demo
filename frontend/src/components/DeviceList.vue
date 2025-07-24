<template>
  <div class="device-list-wrapper">
    <div class="device-header">
      <span>当前在线设备数：<b>{{ data.length }}</b></span>
    </div>
    <div class="batch-bar">
      <el-button type="primary" size="small" :disabled="multipleSelection.length === 0" @click="onBatchSampling">
        {{ batchSamplingText }}
      </el-button>
      <el-button type="primary" size="small" :disabled="multipleSelection.length === 0" @click="onBatchInference" style="margin-left: 12px;">
        {{ batchInferenceText }}
      </el-button>
    </div>
    <el-table
      :data="data"
      class="device-table"
      stripe
      border
      v-loading="loading"
      @selection-change="handleSelectionChange"
      :row-key="row => row.name"
    >
      <el-table-column type="selection" width="48" />
      <el-table-column prop="id" label="ID" width="80" />
      <el-table-column prop="deviceId" label="设备ID" />
      <el-table-column prop="deviceName" label="设备名" />
      <el-table-column label="数据采集开关">
        <template #default="{ row }">
          <el-switch :model-value="row.is_sampling"
            @change="val => onSwitch1(row, val)"
            :loading="switchLoading.value[row.deviceId + '_sampling']"
            active-color="#2563eb" inactive-color="#bcd0f7" />
        </template>
      </el-table-column>
      <el-table-column label="定位服务开关">
        <template #default="{ row }">
          <el-switch :model-value="row.is_inference"
            @change="val => onSwitch2(row, val)"
            :loading="switchLoading.value[row.deviceId + '_inference']"
            active-color="#2563eb" inactive-color="#bcd0f7" />
        </template>
      </el-table-column>
    </el-table>
    <el-empty v-if="!loading && data.length === 0" description="暂无在线设备" />
    <el-alert v-if="error" :title="error" type="error" show-icon style="margin-top:16px;" />
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onUnmounted, reactive } from 'vue'

const data = ref([])
const loading = ref(false)
const error = ref('')
const multipleSelection = ref([])
let statusTimer = null
const switchLoading = reactive({})

const batchSamplingText = computed(() => {
  if (multipleSelection.value.length === 0) return '批量采集';
  const allOn = multipleSelection.value.every(d => d.is_sampling)
  return allOn ? '批量关闭采集' : '批量开启采集'
})
const batchInferenceText = computed(() => {
  if (multipleSelection.value.length === 0) return '批量定位';
  const allOn = multipleSelection.value.every(d => d.is_inference)
  return allOn ? '批量关闭定位' : '批量开启定位'
})

function handleSelectionChange(val) {
  multipleSelection.value = val
}

async function fetchDeviceStatus() {
  try {
    const res = await fetch('/device_status')
    if (!res.ok) return
    const status = await res.json()
    for (const device of data.value) {
      if (status[device.deviceId]) {
        device.is_sampling = status[device.deviceId].is_sampling
        device.is_inference = status[device.deviceId].is_inference
      }
    }
  } catch (e) {}
}

onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await fetch('/devices')
    if (!res.ok) throw new Error('获取设备失败')
    const resData = await res.json()
    if (resData.error) throw new Error(resData.error)
    for (const [i, device] of resData.entries()) {
      data.value.push({
        id: i + 1,
        deviceId: device.deviceId,
        deviceName: device.deviceName,
        is_sampling : false,
        is_inference : false
      })
    }
    await fetchDeviceStatus()
    statusTimer = setInterval(fetchDeviceStatus, 2000)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

onUnmounted(() => {
  if (statusTimer) clearInterval(statusTimer)
})

// 修正后的开关方法
async function onSwitch1(device, newVal) {
  const key = device.deviceId + '_sampling'
  switchLoading[key] = true
  const url = newVal
    ? `/start_sample?target_device_id=${device.deviceId}`
    : `/end_sample?target_device_id=${device.deviceId}`;
  try {
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_device_id: device.deviceId })
    });
    await fetchDeviceStatus();
  } catch (error) {
    alert(error.message)
  } finally {
    switchLoading[key] = false
  }
}

async function onSwitch2(device, newVal) {
  const key = device.deviceId + '_inference'
  switchLoading[key] = true
  const url = newVal
    ? `/start_inference?target_device_id=${device.deviceId}`
    : `/end_inference?target_device_id=${device.deviceId}`;
  try {
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_device_id: device.deviceId })
    });
    await fetchDeviceStatus();
  } catch (error) {
    alert(error.message)
  } finally {
    switchLoading[key] = false
  }
}

// 批量操作
async function onBatchSampling() {
  const allOn = multipleSelection.value.every(d => d.is_sampling)
  for (const device of multipleSelection.value) {
    if (allOn && device.is_sampling) {
      await onSwitch1(device, false)
    } else if (!allOn && !device.is_sampling) {
      await onSwitch1(device, true)
    }
  }
}
async function onBatchInference() {
  const allOn = multipleSelection.value.every(d => d.is_inference)
  for (const device of multipleSelection.value) {
    if (allOn && device.is_inference) {
      await onSwitch2(device, false)
    } else if (!allOn && !device.is_inference) {
      await onSwitch2(device, true)
    }
  }
}
</script>

<style scoped>
.device-list-wrapper {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 4px 24px #2563eb11, 0 1.5px 4px #2563eb22;
  padding: 32px 24px;
  max-width: 800px;
  margin: 0 auto;
}
.device-header {
  font-size: 1.08rem;
  color: #2563eb;
  margin-bottom: 10px;
  font-weight: 500;
}
.batch-bar {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  gap: 8px;
}
.device-table {
  border-radius: 12px;
  overflow: hidden;
  font-size: 1rem;
}
.el-table th {
  background: #e3edfa;
  color: #2563eb;
  font-weight: 600;
}
.el-table__row:hover {
  background: #f0f6ff !important;
}
.el-switch {
  --el-switch-on-color: #2563eb;
  --el-switch-off-color: #bcd0f7;
}
</style> 