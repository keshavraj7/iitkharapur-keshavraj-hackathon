from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, signals, ingest,rebalancer

app = FastAPI(
    title="Financial Risk Intelligence",
    description="AI/NLP engine for financial risk signals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(signals.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(rebalancer.router, prefix="/api/v1")