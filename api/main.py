import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.database import Base, engine, SessionLocal
from api.routes import auth, devices, incidents, messages, ably_token

logger = logging.getLogger(__name__)


def _migrate_add_columns_sqlite() -> None:
    """
    Apply forward-only ADD COLUMN migrations on an existing SQLite DB.

    SQLAlchemy's ``Base.metadata.create_all`` only creates tables that don't
    exist yet — it never alters existing ones, so a model change like adding
    ``Incident.gun_count`` would otherwise break the next query against an
    already-populated dev DB. Listing migrations explicitly here keeps the
    transition seamless without pulling in Alembic for what is currently a
    single-developer dev DB.

    Each migration is gated on the column not already being present so the
    function is idempotent across restarts. Restricted to SQLite because
    other backends should be migrated through Alembic (or equivalent).
    """
    if not engine.url.drivername.startswith("sqlite"):
        return

    migrations: tuple[tuple[str, str, str], ...] = (
        # (table, column, ALTER statement)
        ("incidents", "gun_count",
         "ALTER TABLE incidents ADD COLUMN gun_count INTEGER"),
    )

    with engine.connect() as conn:
        for table, column, ddl in migrations:
            cols = {row[1] for row in conn.exec_driver_sql(
                f"PRAGMA table_info({table})"
            ).fetchall()}
            if column not in cols:
                logger.info("Migrating SQLite: %s -> add column %s", table, column)
                conn.exec_driver_sql(ddl)
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables and seed demo data on startup
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns_sqlite()
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
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
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
