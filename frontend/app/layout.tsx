import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'DataFlow – AI 기반 데이터 사이언스 플랫폼',
  description: 'Next.js + FastAPI로 구축된 데이터 사이언스 SaaS',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <head>
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
        <link rel="manifest" href="/manifest.webmanifest" />
      </head>
      <body>{children}</body>
    </html>
  )
}


