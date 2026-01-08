// sw.js 內容
self.addEventListener('install', (e) => {
  console.log('Service Worker: Installed');
});

self.addEventListener('fetch', (e) => {
  // 保持空攔截，以滿足 PWA 安裝條件
});

