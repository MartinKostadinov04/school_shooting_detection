import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.database import Base, engine, SessionLocal
from api.routes import auth, devices, incidents, messages, ably_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables and seed demo data on startup
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        auth.seed_users(db)
        devices.seed_devices(db)
        incidents.seed_incidents(db)
        messages.seed_messages(db)
    finally:
        db.close()
    yield


app = FastAPI(title="TacticalEye API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all route modules under /api
for router_module in (auth, devices, incidents, messages, ably_token):
    app.include_router(router_module.router, prefix="/api")

# Serve built frontend in production
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="spa")
