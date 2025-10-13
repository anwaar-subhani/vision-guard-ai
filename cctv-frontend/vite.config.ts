import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,  // Your custom port
    open: true,  // Automatically open browser
    host: true,  // Allow external connections
  },
})
