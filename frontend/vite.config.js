import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // The Flask API only allows CORS requests from port 5173, so fail loudly
    // instead of silently switching to another port if 5173 is taken.
    port: 5173,
    strictPort: true,
  },
});
