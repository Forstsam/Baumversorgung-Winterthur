import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages: https://forstsam.github.io/Baumversorgung-Winterthur/
export default defineConfig({
  plugins: [react()],
  base: '/Baumversorgung-Winterthur/',
})
