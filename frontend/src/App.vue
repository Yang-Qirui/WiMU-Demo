<template>
  <div class="layout">
    <el-container style="height: 100vh">
      <aside
        class="sidebar-glass"
        :class="{ 'sidebar-glass--open': sidebarOpen }"
        @mouseenter="sidebarOpen = true"
        @mouseleave="sidebarOpen = false"
      >
        <div class="logo-ios">
          <span v-if="sidebarOpen" class="logo-title">WiMU Demo</span>
        </div>
        <nav class="menu-list">
          <div
            class="menu-item"
            :class="{ active: activeMenu === '1' }"
            @mouseenter="activeMenu = '1'"
          >
            <el-icon><i-ep-iphone /></el-icon>
            <span v-if="sidebarOpen">📱设备管理</span>
          </div>
          <div
            class="menu-item disabled"
            :class="{ active: activeMenu === '2' }"
            @mouseenter="activeMenu = '2'"
          >
            <el-icon><i-ep-data-analysis /></el-icon>
            <span v-if="sidebarOpen">📉数据统计</span>
          </div>
        </nav>
      </aside>
      <el-container>
        <el-header class="header-ios">
          <span class="header-title"><el-icon style="vertical-align:middle;"><i-ep-setting /></el-icon> ⚙️设备管理面板</span>
        </el-header>
        <el-main class="main-content-ios">
          <DeviceList />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import DeviceList from './components/DeviceList.vue'
import { Iphone, DataAnalysis, Setting } from '@element-plus/icons-vue'
const activeMenu = ref('1')
const sidebarOpen = ref(false)
</script>

<style scoped>
.layout {
  background: linear-gradient(135deg, #e0e7ef 0%, #f4f8fb 100%);
  min-height: 100vh;
}
.sidebar-glass {
  width: 64px;
  transition: width 0.25s cubic-bezier(.4,0,.2,1);
  /* 浅蓝色毛玻璃渐变 */
  background: linear-gradient(135deg, rgba(210,230,255,0.75) 60%, rgba(180,210,255,0.55) 100%);
  box-shadow: 2px 0 16px #1e3a5f22;
  backdrop-filter: blur(18px) saturate(1.5);
  border-right: 1.5px solid #e3eaf7cc;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 24px;
  z-index: 10;
  overflow: hidden;
}
.sidebar-glass--open {
  width: 180px;
}
.logo-ios {
  font-size: 2rem;
  color: #2563eb;
  margin-bottom: 36px;
  font-weight: bold;
  letter-spacing: 2px;
  text-align: center;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.logo-emoji {
  font-size: 2.1rem;
  margin-right: 2px;
}
.logo-title {
  font-size: 1.2rem;
  color: #2563eb;
  font-weight: 600;
  letter-spacing: 1px;
}
.menu-list {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.menu-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 18px;
  border-radius: 12px;
  color: #2563eb;
  font-size: 1.08rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.18s, color 0.18s;
  margin: 0 8px;
  opacity: 0.85;
}
.menu-item.active,
.menu-item:hover {
  background: rgba(37,99,235,0.12);
  color: #2563eb;
  opacity: 1;
}
.menu-item.disabled {
  color: #bcd0f7;
  cursor: not-allowed;
  opacity: 0.6;
}
.header-ios {
  background: rgba(255,255,255,0.7);
  color: #2563eb;
  font-size: 1.2rem;
  font-weight: 500;
  padding-left: 24px;
  display: flex;
  align-items: center;
  height: 56px;
  box-shadow: 0 2px 8px #e0e7ef33;
  border-bottom: 1.5px solid #e3eaf7cc;
  backdrop-filter: blur(12px) saturate(1.2);
}
.header-title {
  letter-spacing: 1px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.main-content-ios {
  padding: 32px 24px;
  background: transparent;
  min-height: calc(100vh - 56px);
}
</style> 