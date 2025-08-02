/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'brand-primary': '#004aad',
        'brand-secondary': '#198754',
        'brand-gradient-start': '#0056cc',
        'brand-gradient-end': '#004aad',
        'assistant-gradient-start': '#20C997',
        'assistant-gradient-end': '#14B8A6',
        'card-bg': '#FFFFFF',
        'page-bg': '#F8F9FA',
        'text-primary': '#212529',
        'text-secondary': '#6C757D',
        'murmur': '#DC3545',
        'normal': '#198754',
        'artifact': '#6C757D',
        'extrahls': '#FD7E14',
        'extrasystole': '#FFC107',
        'danger-bg': '#F8D7DA',
        'danger-text': '#842029',
        'danger-border': '#F5C2C7',
      }
    },
  },
  plugins: [],
} 