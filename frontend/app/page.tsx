"use client"

import { useEffect, useMemo, useRef, useState } from 'react'
import { apiDetectTarget, apiEvaluate, apiExplore, apiPredict, apiPreprocess, apiTrain, apiUpload, type DetectTargetResponse, type EvaluateResponse, type ExploreResponse, type PredictResponse, type PreprocessResponse, type TrainResponse, type UploadResponse } from '@/lib/api'

type Step = 'upload' | 'explore' | 'preprocess' | 'target' | 'features' | 'model' | 'train' | 'evaluate' | 'predict' | 'deploy'

const steps: { key: Step; label: string; icon: string; hint: string }[] = [
  { key: 'upload', label: '데이터 업로드', icon: 'fa-upload', hint: 'CSV, Excel, JSON 등' },
  { key: 'explore', label: '데이터 탐색', icon: 'fa-search', hint: '통계 분석 및 시각화' },
  { key: 'preprocess', label: '데이터 전처리', icon: 'fa-cogs', hint: '클렌징 및 변환' },
  { key: 'target', label: '타겟 설정', icon: 'fa-crosshairs', hint: '예측할 변수 선택' },
  { key: 'features', label: '피처 엔지니어링', icon: 'fa-puzzle-piece', hint: '중요 변수 선택/생성' },
  { key: 'model', label: '모델 선택', icon: 'fa-robot', hint: 'AI 알고리즘 선택' },
  { key: 'train', label: '모델 학습', icon: 'fa-dumbbell', hint: '데이터로 모델 훈련' },
  { key: 'evaluate', label: '성능 평가', icon: 'fa-chart-line', hint: '모델 성능 측정' },
  { key: 'predict', label: '예측 수행', icon: 'fa-magic', hint: '새로운 데이터 예측' },
  { key: 'deploy', label: '모델 배포', icon: 'fa-rocket', hint: 'API 및 웹 서비스' },
]

export default function Page() {
  const [current, setCurrent] = useState<Step>('upload')
  const [session, setSession] = useState<string | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [exploreStats, setExploreStats] = useState<ExploreResponse | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [preprocessResult, setPreprocessResult] = useState<PreprocessResponse | null>(null)
  const [target, setTarget] = useState<string>('')
  const [problemType, setProblemType] = useState<string>('')
  const [classDist, setClassDist] = useState<Record<string, number> | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('Random Forest')
  const [training, setTraining] = useState(false)
  const [trainEpoch, setTrainEpoch] = useState(0)
  const [trainMetrics, setTrainMetrics] = useState<TrainResponse | null>(null)
  const [evalMetrics, setEvalMetrics] = useState<EvaluateResponse | null>(null)
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  // Header counters demo
  const [datasetsProcessed, modelsTrained, predictionsMade] = useMemo(() => {
    return [
      Math.floor(Math.random() * 50) + 25,
      Math.floor(Math.random() * 20) + 10,
      Math.floor(Math.random() * 1000) + 500,
    ]
  }, [current])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter') {
        const idx = steps.findIndex(s => s.key === current)
        if (idx < steps.length - 1) setCurrent(steps[idx + 1].key)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [current])

  async function onUpload(file: File) {
    try {
      setUploading(true)
      setProgress(0)
      const timer = setInterval(() => setProgress(prev => Math.min(prev + 8, 98)), 120)
      const res = await apiUpload(file)
      setSession(res.session)
      setColumns(res.columns)
      setExploreStats(res)
      clearInterval(timer)
      setProgress(100)
      setTimeout(() => setUploading(false), 300)
      setCurrent('explore')
    } catch (e) {
      setUploading(false)
      alert('업로드 실패: ' + (e as Error).message)
    }
  }

  async function refreshExplore() {
    if (!session) return
    const res = await apiExplore(session)
    setExploreStats(res)
  }

  async function runPreprocess() {
    if (!session) return
    const options = {
      impute_missing: true,
      remove_outliers: true,
      normalize: false,
      standardize: false,
      encode_categorical: true,
    }
    const res = await apiPreprocess(session, options)
    setPreprocessResult(res)
  }

  async function onDetectTarget() {
    if (!session || !target) return
    const res: DetectTargetResponse = await apiDetectTarget(session, target)
    setProblemType(res.problem_type)
    setClassDist(res.class_distribution || null)
  }

  async function onTrain() {
    if (!session) return
    setTraining(true)
    setTrainEpoch(0)
    setTrainMetrics(null)
    const maxEpoch = 50
    const ticker = setInterval(() => setTrainEpoch(prev => Math.min(prev + 1, maxEpoch)), 100)
    const res: TrainResponse = await apiTrain(session, { model_name: selectedModel })
    setTrainMetrics(res)
    clearInterval(ticker)
    setTraining(false)
  }

  async function onEvaluate() {
    if (!session) return
    const res: EvaluateResponse = await apiEvaluate(session)
    setEvalMetrics(res)
  }

  async function onPredict(form: FormData) {
    if (!session) return
    const features: Record<string, any> = {}
    columns.filter(c => c !== target).forEach(c => {
      const v = form.get(c)
      features[c] = v === null ? null : (v as string)
    })
    const res: PredictResponse = await apiPredict(session, features)
    setPredictResult(res)
  }

  return (
    <div className="container mx-auto max-w-[1400px] p-5">
      {/* Header */}
      <div className="card-glass flex items-center justify-between p-6 mb-6">
        <div className="text-primary text-2xl font-bold flex items-center gap-3">
          <i className="fas fa-brain" />
          DataFlow
        </div>
        <div className="flex items-center gap-6">
          <div className="text-center">
            <div className="text-primary text-xl font-bold" id="datasetsProcessed">{datasetsProcessed}</div>
            <div className="text-xs text-gray-600 uppercase">데이터셋 처리</div>
          </div>
          <div className="text-center">
            <div className="text-primary text-xl font-bold" id="modelsTrained">{modelsTrained}</div>
            <div className="text-xs text-gray-600 uppercase">모델 학습</div>
          </div>
          <div className="text-center">
            <div className="text-primary text-xl font-bold" id="predictionsMADE">{predictionsMade.toLocaleString()}</div>
            <div className="text-xs text-gray-600 uppercase">예측 수행</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6 h-[calc(100vh-150px)]">
        {/* Sidebar */}
        <aside className="card-glass overflow-y-auto p-6">
          <ul className="space-y-3">
            {steps.map(s => {
              const isActive = current === s.key
              return (
                <li key={s.key}>
                  <button
                    onClick={() => setCurrent(s.key)}
                    className={`w-full flex items-center gap-3 rounded-xl border-2 transition-all px-3 py-3 ${isActive ? 'bg-gradient-to-br from-primary to-secondary text-white border-primary shadow' : 'hover:bg-primary/10 border-transparent'}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isActive ? 'bg-white/20' : 'bg-primary/20 text-primary'}`}>
                      <i className={`fas ${s.icon}`} />
                    </div>
                    <div className="text-left">
                      <div className="font-semibold">{s.label}</div>
                      <small className="text-gray-600">{s.hint}</small>
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>
        </aside>

        {/* Workspace */}
        <main className="card-glass p-6 overflow-y-auto">
          {/* Upload */}
          {current === 'upload' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">데이터 업로드</h2>
              <div
                className="upload-zone"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault() }}
                onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onUpload(f) }}
              >
                <div className="text-5xl text-primary mb-3"><i className="fas fa-cloud-upload-alt"/></div>
                <div className="text-lg text-gray-700">파일을 드래그하여 업로드하거나 클릭하여 선택</div>
                <div className="text-sm text-gray-500">최대 100MB까지 지원</div>
                <input ref={fileInputRef} type="file" hidden accept=".csv,.xlsx,.xls,.json,.txt" onChange={e => e.target.files && onUpload(e.target.files[0])} />
              </div>
              {uploading && (
                <div className="mt-5">
                  <div className="w-full h-2 rounded bg-gray-200 overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-primary to-emerald-500 transition-all" style={{ width: `${progress}%` }} />
                  </div>
                  <p className="mt-2 text-gray-700">업로드 중... {progress}%</p>
                </div>
              )}
            </section>
          )}

          {/* Explore */}
          {current === 'explore' && (
            <section className="section-enter">
              <div className="flex items-center justify-between mb-5 pb-5 border-b">
                <h2 className="text-2xl font-bold text-gray-800">데이터 탐색 및 분석</h2>
                <button className="btn-secondary px-4 py-2" onClick={refreshExplore}><i className="fas fa-rotate"/> 새로고침</button>
              </div>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center">
                <i className="fas fa-info-circle"/> 데이터의 기본 통계와 분포를 확인하세요.
              </div>
              {exploreStats && (
                <>
                  <div className="grid md:grid-cols-4 gap-4 mt-6">
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{exploreStats.total_rows}</div><div className="text-sm text-gray-600">총 행 수</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{exploreStats.total_cols}</div><div className="text-sm text-gray-600">총 열 수</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{exploreStats.missing_ratio}%</div><div className="text-sm text-gray-600">결측값 비율</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{exploreStats.outliers}</div><div className="text-sm text-gray-600">이상치 개수</div></div>
                  </div>
                  <div className="mt-6 bg-slate-50 rounded-xl p-5 overflow-x-auto">
                    <h3 className="font-semibold mb-3">데이터 미리보기</h3>
                    <table className="w-full text-sm">
                      <thead className="bg-slate-100">
                        <tr>
                          {exploreStats.columns.map((c: string) => (<th key={c} className="text-left p-2 text-slate-600">{c}</th>))}
                        </tr>
                      </thead>
                      <tbody>
                        {exploreStats.preview.map((row: any, idx: number) => (
                          <tr key={idx} className="border-b">
                            {exploreStats.columns.map((c: string) => (<td key={c} className="p-2">{String(row[c])}</td>))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </section>
          )}

          {/* Preprocess */}
          {current === 'preprocess' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">데이터 전처리</h2>
              <div className="rounded-xl p-4 bg-amber-50 text-amber-800 border-l-4 border-amber-400 flex gap-3 items-center mb-5">
                <i className="fas fa-exclamation-triangle"/> 데이터 품질을 향상시키기 위해 전처리 옵션을 선택하세요.
              </div>
              <div className="space-y-3 mb-4">
                <label className="font-medium text-gray-700 flex items-center gap-2"><input type="checkbox" defaultChecked /> 결측값 처리 (평균/최빈값)</label>
                <label className="font-medium text-gray-700 flex items-center gap-2"><input type="checkbox" defaultChecked /> 이상치 제거 (IQR)</label>
                <label className="font-medium text-gray-700 flex items-center gap-2"><input type="checkbox" /> 데이터 정규화 (Min-Max)</label>
                <label className="font-medium text-gray-700 flex items-center gap-2"><input type="checkbox" /> 데이터 표준화 (Z-Score)</label>
                <label className="font-medium text-gray-700 flex items-center gap-2"><input type="checkbox" defaultChecked /> 범주형 변수 인코딩</label>
              </div>
              <button className="btn-primary px-5 py-2" onClick={runPreprocess}><i className="fas fa-play"/> 전처리 실행</button>
              {preprocessResult && (
                <div className="mt-5">
                  <div className="rounded-xl p-4 bg-emerald-50 text-emerald-800 border-l-4 border-emerald-500 flex gap-3 items-center"><i className="fas fa-check-circle"/> 전처리가 성공적으로 완료되었습니다!</div>
                  <div className="grid md:grid-cols-4 gap-4 mt-4">
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{preprocessResult.processed_rows}</div><div className="text-sm text-gray-600">처리된 행 수</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{preprocessResult.final_features}</div><div className="text-sm text-gray-600">최종 피처 수</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">{(preprocessResult.missing_ratio * 100).toFixed(0)}%</div><div className="text-sm text-gray-600">결측값 비율</div></div>
                    <div className="metric-card"><div className="text-primary text-2xl font-bold">100%</div><div className="text-sm text-gray-600">데이터 품질</div></div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Target */}
          {current === 'target' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">타겟 변수 설정</h2>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center mb-5">
                <i className="fas fa-info-circle"/> 예측하고자 하는 타겟 변수를 선택하세요. 문제 유형이 자동으로 감지됩니다.
              </div>
              <div className="mb-4">
                <label className="block font-semibold text-gray-700 mb-2">타겟 변수 선택</label>
                <select value={target} onChange={e => setTarget(e.target.value)} className="w-full border-2 rounded-lg p-3">
                  <option value="">타겟 변수를 선택하세요</option>
                  {columns.map(c => (<option key={c} value={c}>{c}</option>))}
                </select>
              </div>
              <div className="mb-4">
                <label className="block font-semibold text-gray-700 mb-2">문제 유형</label>
                <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400">{problemType ? `${problemType === 'classification' ? '분류' : '회귀'} 문제로 감지되었습니다.` : '타겟 변수를 선택하면 자동으로 감지됩니다.'}</div>
              </div>
              <button disabled={!target} className="btn-primary px-5 py-2 disabled:opacity-50" onClick={onDetectTarget}><i className="fas fa-wand-magic-sparkles"/> 자동 감지</button>
              {classDist && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-5">
                  {Object.entries(classDist).map(([k, v]) => (
                    <div key={k} className="metric-card"><div className="text-primary text-2xl font-bold">{v}</div><div className="text-sm text-gray-600">클래스 {k}</div></div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Features (static illustrative) */}
          {current === 'features' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">피처 엔지니어링</h2>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center mb-5">
                <i className="fas fa-info-circle"/> 중요한 피처를 선택하거나 새로운 피처를 생성하여 모델 성능을 향상시키세요.
              </div>
              <div className="space-y-3">
                {[{n:'income', s:0.85},{n:'age', s:0.72},{n:'experience', s:0.68},{n:'education', s:0.45},{n:'location', s:0.32}].map(f => (
                  <div key={f.n} className="flex items-center gap-3">
                    <div className="w-40 font-medium">{f.n}</div>
                    <div className="flex-1 h-2 bg-gray-200 rounded">
                      <div className="h-full rounded bg-gradient-to-r from-primary to-secondary" style={{ width: `${Math.round(f.s*100)}%` }}/>
                    </div>
                    <div className="w-12 text-right text-primary font-semibold">{f.s.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Model */}
          {current === 'model' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">AI 모델 선택</h2>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center mb-5">
                <i className="fas fa-info-circle"/> 데이터와 문제 유형에 적합한 AI 모델을 선택하세요.
              </div>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {["Random Forest","XGBoost","Neural Network","Support Vector Machine","Linear Regression","Logistic Regression","AutoML","Transformer"].map(m => (
                  <button key={m} onClick={() => setSelectedModel(m)} className={`text-left border-2 rounded-xl p-4 transition ${selectedModel===m ? 'border-primary bg-primary/5' : 'hover:-translate-y-0.5'}`}>
                    <div className="font-bold">{m}</div>
                    <div className="text-sm text-gray-600 mt-1">{m==='Random Forest'?'앙상블 학습으로 높은 정확도와 해석 가능성': '모델 설명 텍스트'}</div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {m==='Random Forest' && (<><span className="text-xs bg-indigo-100 text-primary px-2 py-0.5 rounded">분류</span><span className="text-xs bg-indigo-100 text-primary px-2 py-0.5 rounded">회귀</span></>)}
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Train */}
          {current === 'train' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">모델 학습</h2>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center mb-5">
                <i className="fas fa-info-circle"/> 선택한 모델을 데이터로 학습시킵니다.
              </div>
              <button className="btn-primary px-5 py-2" onClick={onTrain} disabled={training}><i className="fas fa-play"/> 모델 학습 시작</button>
              <div className="mt-6">
                <h3 className="font-semibold">학습 진행 상황</h3>
                <div className="w-full h-2 rounded bg-gray-200 overflow-hidden mt-2">
                  <div className="h-full bg-gradient-to-r from-primary to-emerald-500 transition-all" style={{ width: `${(trainEpoch/50)*100}%` }} />
                </div>
                <div className="flex justify-between text-sm text-gray-700 mt-2">
                  <span>Epoch {trainEpoch}/50</span>
                  <span>Loss: {(2 - trainEpoch * 0.03).toFixed(4)}</span>
                </div>
              </div>
              {trainMetrics && (
                <div className="mt-5 rounded-xl p-4 bg-emerald-50 text-emerald-800 border-l-4 border-emerald-500 flex gap-3 items-center"><i className="fas fa-check-circle"/> 모델 학습이 완료되었습니다!</div>
              )}
            </section>
          )}

          {/* Evaluate */}
          {current === 'evaluate' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">모델 성능 평가</h2>
              <div className="rounded-xl p-4 bg-emerald-50 text-emerald-800 border-l-4 border-emerald-500 flex gap-3 items-center mb-5"><i className="fas fa-check-circle"/> 모델 학습이 완료되었습니다. 다양한 지표로 성능을 평가해보세요.</div>
              <button className="btn-secondary px-4 py-2" onClick={onEvaluate}><i className="fas fa-chart-line"/> 평가 실행</button>
              {evalMetrics && (
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mt-5">
                  {Object.entries(evalMetrics.metrics).map(([k, v]) => (
                    <div key={k} className="metric-card"><div className="text-primary text-2xl font-bold">{typeof v === 'number' ? (k==='auc'? v.toFixed(2) : (v>=1? (v*100).toFixed(1)+'%' : (v*100).toFixed(1)+'%')) : '-'}</div><div className="text-sm text-gray-600">{k.toUpperCase()}</div></div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Predict */}
          {current === 'predict' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">예측 수행</h2>
              <div className="rounded-xl p-4 bg-blue-50 text-blue-800 border-l-4 border-blue-400 flex gap-3 items-center mb-5"><i className="fas fa-info-circle"/> 학습된 모델로 새로운 데이터에 대한 예측을 수행하세요.</div>
              <form
                className="bg-slate-50 rounded-xl p-5"
                onSubmit={(e) => { e.preventDefault(); const fd = new FormData(e.currentTarget); onPredict(fd) }}
              >
                <div className="grid sm:grid-cols-2 gap-4">
                  {columns.filter(c => c !== target).slice(0, 8).map(c => (
                    <div key={c}>
                      <label className="block font-semibold text-gray-700 mb-1">{c}</label>
                      <input name={c} className="w-full border-2 rounded-lg p-3" placeholder={c} />
                    </div>
                  ))}
                </div>
                <button className="btn-primary px-5 py-2 mt-4" type="submit"><i className="fas fa-magic"/> 예측 수행</button>
              </form>
              {predictResult && (
                <div className="mt-5 text-center text-white rounded-xl p-8 bg-gradient-to-br from-primary to-secondary">
                  <div className="text-3xl font-bold mb-2">{String(predictResult.prediction)}</div>
                  {predictResult.probabilities && <div className="opacity-90">예측 확률: [{predictResult.probabilities.map((p:number)=>p.toFixed(3)).join(', ')}]</div>}
                </div>
              )}
            </section>
          )}

          {/* Deploy */}
          {current === 'deploy' && (
            <section className="section-enter">
              <h2 className="text-2xl font-bold text-gray-800 mb-5">모델 배포 및 서비스화</h2>
              <div className="rounded-xl p-4 bg-emerald-50 text-emerald-800 border-l-4 border-emerald-500 flex gap-3 items-center mb-5"><i className="fas fa-rocket"/> 모델을 실제 서비스에서 사용할 수 있도록 배포하세요.</div>
              <div className="bg-slate-50 rounded-xl p-5">
                <h3 className="font-semibold mb-2">API 사용 예제</h3>
                <pre className="bg-slate-900 text-slate-200 text-xs p-4 rounded-md overflow-x-auto">{`curl -X POST ${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}/predict?session=${session || '<session>'} \
-H "Content-Type: application/json" \
-d '{"features": {"age": 30, "income": 55000}}'`}</pre>
              </div>
            </section>
          )}
        </main>
      </div>
      {/* fontawesome cdn */}
      <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet" />
    </div>
  )
}


