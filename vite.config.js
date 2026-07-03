import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' erzeugt relative Pfade -> funktioniert sowohl auf Netlify/Vercel (Wurzel)
// als auch auf GitHub Pages (Unterpfad /Baumversorgung-Winterthur/), ohne Änderung.
export default defineConfig({
  base: './',
  plugins: [react()],
})
