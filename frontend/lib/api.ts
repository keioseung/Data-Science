export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

export type UploadResponse = {
  session: string
  total_rows: number
  total_cols: number
  missing_ratio: number
  outliers: number
  preview: any[]
  columns: string[]
}

export type ExploreResponse = Omit<UploadResponse, 'session'>

export type PreprocessResponse = {
  processed_rows: number
  final_features: number
  missing_ratio: number
  data_quality: number
}

export type DetectTargetResponse = {
  problem_type: 'classification' | 'regression'
  class_distribution?: Record<string, number> | null
}

export type TrainResponse = {
  trained: boolean
  problem_type: 'classification' | 'regression'
  metrics: Record<string, number | null>
  test_size: number
  model_name: string
}

export type EvaluateResponse = {
  problem_type: 'classification' | 'regression'
  metrics: Record<string, number | null>
}

export type PredictResponse = {
  prediction: string | number
  probabilities?: number[]
}

export async function apiUpload(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  return request<UploadResponse>(`/upload`, { method: 'POST', body: form })
}

export async function apiExplore(session: string): Promise<ExploreResponse> {
  return request<ExploreResponse>(`/explore?session=${encodeURIComponent(session)}`)
}

export async function apiPreprocess(session: string, options: any): Promise<PreprocessResponse> {
  return request<PreprocessResponse>(`/preprocess?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  })
}

export async function apiDetectTarget(session: string, target_variable: string): Promise<DetectTargetResponse> {
  return request<DetectTargetResponse>(`/target/detect?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_variable }),
  })
}

export async function apiTrain(session: string, payload: any): Promise<TrainResponse> {
  return request<TrainResponse>(`/train?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function apiEvaluate(session: string): Promise<EvaluateResponse> {
  return request<EvaluateResponse>(`/evaluate?session=${encodeURIComponent(session)}`)
}

export async function apiPredict(session: string, features: Record<string, any>): Promise<PredictResponse> {
  return request<PredictResponse>(`/predict?session=${encodeURIComponent(session)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })
}


