import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Compose sets VITE_PROXY_TARGET=http://backend:4000; on a bare host the backend
// is reachable on localhost. Read at config time (Node), never in app code.
const proxyTarget = process.env.VITE_PROXY_TARGET || 'http://localhost:4000';

const proxyEntry = {
  target: proxyTarget,
  changeOrigin: true,
};

// Vite rejects requests whose Host header it does not recognise. That check exists to stop
// DNS rebinding (a hostile page resolving its own domain to 127.0.0.1 and then reading your
// dev server), so the fix for "Blocked request. This host is not allowed" is to name the
// hosts you actually serve on -- not to switch the check off.
//
// A leading dot matches the domain and every subdomain, so these entries cover the
// throwaway hostnames the usual dev tunnels hand out without opening the door to anything
// else. They are safe to allow by default precisely because they resolve to the provider's
// edge, never to a victim's loopback.
const TUNNEL_HOSTS = [
  '.trycloudflare.com', // cloudflared tunnel --url
  '.ngrok-free.app',
  '.ngrok.io',
  '.ngrok.app',
  '.loca.lt', // localtunnel
  '.devtunnels.ms', // VS Code / Dev Tunnels
  '.github.dev', // Codespaces
  '.gitpod.io',
  '.repl.co',
  '.csb.app', // CodeSandbox
];

// VITE_ALLOWED_HOSTS: comma-separated extra hostnames, or `*` to allow any host.
// `*` disables the rebinding protection entirely -- only reasonable on a trusted network.
const allowedHostsEnv = (process.env.VITE_ALLOWED_HOSTS || '').trim();
const allowedHosts =
  allowedHostsEnv === '*' || allowedHostsEnv === 'true'
    ? true
    : [
        ...TUNNEL_HOSTS,
        ...allowedHostsEnv
          .split(',')
          .map((host) => host.trim())
          .filter(Boolean),
      ];

// Behind an HTTPS tunnel the page arrives on 443, but the HMR client would still dial the
// origin port (5173) over plain ws and fail. Point it at the public port instead:
//   VITE_HMR_CLIENT_PORT=443 VITE_HMR_PROTOCOL=wss
// Left unset for normal local dev, where Vite's own defaults are already correct.
const hmrClientPort = Number.parseInt(process.env.VITE_HMR_CLIENT_PORT || '', 10);
const hmrProtocol = (process.env.VITE_HMR_PROTOCOL || '').trim();
const hmr =
  Number.isInteger(hmrClientPort) || hmrProtocol
    ? {
        ...(Number.isInteger(hmrClientPort) ? { clientPort: hmrClientPort } : {}),
        ...(hmrProtocol ? { protocol: hmrProtocol } : {}),
      }
    : undefined;

export default defineConfig({
  plugins: [react()],
  server: {
    // host:true binds 0.0.0.0 so the port is reachable from outside the container.
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts,
    ...(hmr ? { hmr } : {}),
    // Bind mounts on Docker Desktop / some Linux setups do not deliver inotify
    // events, so polling is the only reliable way to get HMR inside the container.
    watch: { usePolling: true },
    proxy: {
      '/api': proxyEntry,
      // Model weights are served by the backend; proxying keeps every fetch
      // same-origin so no CORS preflight is needed in the dev server.
      '/models': proxyEntry,
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts,
  },
  build: {
    // @tensorflow/tfjs alone is well over the default 500 kB warning threshold.
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/@tensorflow')) return 'tfjs';
          if (
            id.includes('node_modules/react-dom') ||
            id.includes('node_modules/react/') ||
            id.includes('node_modules/scheduler')
          ) {
            return 'react-vendor';
          }
          return undefined;
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.{test,spec}.{js,jsx}', 'src/**/__tests__/**/*.{js,jsx}'],
    restoreMocks: true,
  },
});
