import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './pages/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4f46e5',
          600: '#4f46e5',
          700: '#4338ca'
        },
        secondary: '#7c3aed'
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0, 0, 0, 0.1)'
      },
      backdropBlur: {
        xs: '2px'
      }
    }
  },
  plugins: []
}

export default config


