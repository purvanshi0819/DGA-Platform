/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#08131f", 900: "#0b1b2b", 800: "#12283d", 700: "#1b3a56", 600: "#2a4d6e" },
        brand: { 50: "#e8fafa", 100: "#c9f2f2", 400: "#2fc4c4", 500: "#14a8a8", 600: "#0e8a8c", 700: "#0d6f72" },
        amber: { 50: "#fff8e6", 100: "#ffefc2", 500: "#e8a317", 600: "#c4860b", 700: "#9a6808" },
        surface: "#f3f6f9",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"], mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
      boxShadow: { card: "0 1px 2px rgba(11,27,43,.05), 0 1px 3px rgba(11,27,43,.04)" },
    },
  },
  plugins: [],
};
