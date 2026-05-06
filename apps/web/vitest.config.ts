/**
 * Vitest config for apps/web.
 *
 * Minimal happy-path: `@vitejs/plugin-react` for JSX/TSX transforms,
 * jsdom for DOM globals (so RTL's render/screen work), and a setup
 * file that wires `@testing-library/jest-dom` matchers onto Vitest's
 * `expect`. The `@/` path alias mirrors the Next.js tsconfig so
 * imports written `from "@/lib/submission"` resolve identically in
 * tests and in production.
 */
import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["__tests__/**/*.test.{ts,tsx}"],
    globals: false,
  },
});
