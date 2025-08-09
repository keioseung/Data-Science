export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json()
}

export async function apiUpload(file: File) {
  const form = new FormData()
  form.append('file', file)
  return request<{
    session: string
    total_rows: number
    total_cols: number
    missing_ratio: number
    outliers: number
    preview: any[]
    columns: string[]
  }>(`/upload`, { method: 'POST', body: form })
}

export async function apiExplore(session: string) {
  return request(`/explore?session=${encodeURIComponent(session)}`)
}

export async function apiPreprocess(session: string, options: any) {
  return request(`/preprocess?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  })
}

export async function apiDetectTarget(session: string, target_variable: string) {
  return request(`/target/detect?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_variable }),
  })
}

export async function apiTrain(session: string, payload: any) {
  return request(`/train?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function apiEvaluate(session: string) {
  return request(`/evaluate?session=${encodeURIComponent(session)}`)
}

export async function apiPredict(session: string, features: Record<string, any>) {
  return request(`/predict?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })
}


