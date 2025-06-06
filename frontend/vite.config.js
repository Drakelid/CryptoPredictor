import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      // Fix Plotly import issues
      'plotly.js/dist/plotly': path.resolve(__dirname, 'node_modules/plotly.js-dist/plotly.js'),
      'plotly.js': path.resolve(__dirname, 'node_modules/plotly.js-dist')
    },
  },
  optimizeDeps: {
    include: ['plotly.js-dist']
  }
})
