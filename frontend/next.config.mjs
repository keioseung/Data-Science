/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: [
      'react',
      'react-dom'
    ]
  },
  env: {
    // Used by edge proxy when NEXT_PUBLIC_API_BASE_URL is not set
    BACKEND_URL: process.env.NEXT_PUBLIC_API_BASE_URL
  }
}

export default nextConfig


