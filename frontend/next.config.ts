import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: [
      'react',
      'react-dom'
    ]
  }
}

export default nextConfig


