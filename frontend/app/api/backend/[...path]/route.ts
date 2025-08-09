import { NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'host',
  'content-length',
  'accept-encoding'
])

function getBackendBase() {
  return process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
}

async function proxy(request: Request, context: { params: { path: string[] } }) {
  const { path } = context.params
  const base = getBackendBase().replace(/\/$/, '')
  const targetPath = '/' + (path || []).join('/')
  const url = new URL(request.url)
  const qs = url.search
  const targetUrl = base + targetPath + qs

  const headers: Record<string, string> = {}
  request.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers[key] = value
  })

  const init: RequestInit = {
    method: request.method,
    headers
  }

  if (!['GET', 'HEAD'].includes(request.method)) {
    const bodyBuffer = await request.arrayBuffer()
    init.body = bodyBuffer as any
  }

  const res = await fetch(targetUrl, init)
  const resHeaders = new Headers(res.headers)
  // remove hop-by-hop
  HOP_BY_HOP.forEach(h => resHeaders.delete(h))
  const data = await res.arrayBuffer()
  return new NextResponse(data, { status: res.status, headers: resHeaders })
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as PATCH, proxy as DELETE }
export function OPTIONS() { return NextResponse.json({ ok: true }) }


