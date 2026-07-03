# Baumversorgung Winterthur — Dashboard

Interaktives Stadtbaum-Dashboard (React + Vite). Wird als statische Website
über GitHub Pages veröffentlicht — Betrachter brauchen nur den Link, keine Installation.

## Automatisch veröffentlichen (empfohlen, ohne lokale Installation)
1. Diese Dateien ins GitHub-Repo legen (Repo-Name muss `Baumversorgung-Winterthur` sein,
   sonst `base` in `vite.config.js` anpassen).
2. Im Repo unter **Settings → Pages → Build and deployment → Source**: **GitHub Actions** wählen.
3. Nach jedem Push auf `main` baut GitHub die Seite und veröffentlicht sie unter
   `https://<benutzername>.github.io/Baumversorgung-Winterthur/`.

## Lokal entwickeln (optional, benötigt Node.js)
```
npm install
npm run dev      # Vorschau unter http://localhost:5173
npm run build    # erzeugt den Ordner dist/
```
