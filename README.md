DataFlow SaaS – Next.js 14 + FastAPI (Railway-ready)

Overview
- Frontend: Next.js 14 (App Router, TypeScript, TailwindCSS)
- Backend: FastAPI (pandas + scikit-learn for real processing)
- Deploy: Railway (two services with Dockerfiles)

Monorepo Structure
- frontend/ – Next.js 14 app
- backend/ – FastAPI app

Quickstart (Local)
1) Backend
- cd backend
- python -m venv .venv && source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
- pip install -r requirements.txt
- cp ../.env.example ../.env (and update values as needed)
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

2) Frontend
- cd frontend
- npm install
- copy ../.env.example to .env.local and set NEXT_PUBLIC_API_BASE_URL (e.g. http://localhost:8000)
- npm run dev

Environment Variables
- NEXT_PUBLIC_API_BASE_URL – URL of the FastAPI service (e.g. http://localhost:8000 in dev, Railway backend URL in prod)
- BACKEND_CORS_ORIGINS – Comma-separated allowed origins for CORS (e.g. http://localhost:3000,https://your-frontend.railway.app)

Deploy to Railway
Create two services from this repository:
- Service 1: backend (use backend/Dockerfile, expose 8000)
- Service 2: frontend (use frontend/Dockerfile, expose 3000)

Set Environment Variables on Railway
- Backend service:
  - BACKEND_CORS_ORIGINS: https://<your-frontend>.up.railway.app
- Frontend service:
  - NEXT_PUBLIC_API_BASE_URL: https://<your-backend>.up.railway.app

Notes
- The backend provides production-ready endpoints for upload, preprocessing, training, evaluation, and prediction with scikit-learn. Models are kept per project context. For production persistence, add a database or mounted volume.
- The frontend mirrors and elevates the original data.html flow with a polished UI.


