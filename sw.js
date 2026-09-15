const C="predict-edge-v3.1";const A=["./","./index.html","./manifest.webmanifest","./icon.svg"];
self.addEventListener("install",e=>e.waitUntil(caches.open(C).then(c=>c.addAll(A))));
self.addEventListener("activate",e=>e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==C).map(x=>caches.delete(x))))));
self.addEventListener("fetch",e=>{if(e.request.url.includes("data-api.crypto.com"))return;e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)))});
