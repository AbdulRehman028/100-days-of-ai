export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        abyss: "#070b16",
        steel: "#111a2d",
        card: "#0f1728",
        pulse: "#24d5ff",
        mint: "#22e3a2",
        amber: "#ffc857",
        rose: "#ff7a90",
      },
      fontFamily: {
        display: ["Outfit", "sans-serif"],
        body: ["Sora", "sans-serif"],
        mono: ["Space Grotesk", "monospace"],
      },
      boxShadow: {
        neon: "0 12px 40px rgba(36, 213, 255, 0.18)",
      },
    },
  },
  plugins: [],
};
