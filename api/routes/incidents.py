from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db
from api import models, schemas

router = APIRouter()


def _pad(n: int, width: int = 2) -> str:
    return str(n).zfill(width)


def _today_prefix() -> str:
    d = datetime.now(timezone.utc)
    return f"INC-{d.year}{_pad(d.month)}{_pad(d.day)}"


def _next_incident_id(db: Session) -> str:
    prefix = _today_prefix()
    existing = db.query(models.Incident.id).filter(
        models.Incident.id.like(f"{prefix}-%")
    ).all()
    if not existing:
        return f"{prefix}-001"
    max_seq = max(
        int(row.id.rsplit("-", 1)[-1])
        for row in existing
        if row.id.rsplit("-", 1)[-1].isdigit()
    )
    return f"{prefix}-{_pad(max_seq + 1, 3)}"


def _timeline_out(t: models.IncidentTimeline) -> schemas.TimelineEntryOut:
    return schemas.TimelineEntryOut(
        id=str(t.id),
        timestamp=t.timestamp.isoformat() if t.timestamp else datetime.now(timezone.utc).isoformat(),
        label=t.label,
        detail=t.detail,
    )


def _incident_out(inc: models.Incident) -> schemas.IncidentOut:
    return schemas.IncidentOut(
        id=inc.id,
        createdAt=inc.created_at.isoformat() if inc.created_at else datetime.now(timezone.utc).isoformat(),
        location=inc.location,
        type=inc.type,
        source=inc.source,
        status=inc.status,
        severity=inc.severity,
        description=inc.description,
        probability=inc.probability,
        audioUrl=inc.audio_url,
        videoUrl=inc.video_url,
        videoConfirmed=inc.video_confirmed,
        reportedBy=inc.reported_by,
        timeline=[_timeline_out(t) for t in inc.timeline],
    )


def seed_incidents(db: Session) -> None:
    """Seed the 5 demo incidents from mockData.ts if the DB is empty."""
    if db.query(models.Incident).count() > 0:
        return
    now = datetime.now(timezone.utc)
    min_ = 60
    hour = 60 * min_
    day  = 24 * hour

    from datetime import timedelta
    seeds = [
        {
            "id": "INC-20260421-004",
            "created_at": now - timedelta(seconds=4 * min_),
            "location": "Gymnasium",
            "type": "Gunshot",
            "source": "AUDIO-AI",
            "status": "NEW",
            "severity": "Critical",
            "probability": 0.92,
            "timeline": [
                ("AUDIO-AI detection", "Gunshot probability 0.92 at Gymnasium MIC-GY-01",
                 now - timedelta(seconds=4 * min_)),
                ("Snippet attached", "12s audio snippet captured",
                 now - timedelta(seconds=3 * min_)),
            ],
        },
        {
            "id": "INC-20260421-003",
            "created_at": now - timedelta(seconds=2 * hour),
            "location": "Cafeteria",
            "type": "Suspicious Activity",
            "source": "MANUAL",
            "status": "ACKNOWLEDGED",
            "severity": "High",
            "description": "Two unidentified individuals near west entrance.",
            "reported_by": "school@demo.com",
            "timeline": [
                ("Manual report", "School staff filed structured report",
                 now - timedelta(seconds=2 * hour)),
                ("Acknowledged", "Dispatcher acknowledged",
                 now - timedelta(seconds=2 * hour - 4 * min_)),
            ],
        },
        {
            "id": "INC-20260420-002",
            "created_at": now - timedelta(seconds=day),
            "location": "Main Entrance",
            "type": "Gunshot",
            "source": "AUDIO-AI",
            "status": "RESOLVED",
            "severity": "Medium",
            "probability": 0.61,
            "timeline": [
                ("AUDIO-AI detection", "False positive — fireworks",
                 now - timedelta(seconds=day)),
                ("Resolved", "Marked false positive after on-site verification",
                 now - timedelta(seconds=day - 8 * min_)),
            ],
        },
        {
            "id": "INC-20260419-001",
            "created_at": now - timedelta(seconds=2 * day),
            "location": "Cafeteria",
            "type": "Fire",
            "source": "MANUAL",
            "status": "RESOLVED",
            "severity": "Low",
            "description": "Smoke from kitchen — false alarm.",
            "reported_by": "school@demo.com",
            "timeline": [
                ("Manual report", None, now - timedelta(seconds=2 * day)),
                ("Resolved", None, now - timedelta(seconds=2 * day - 12 * min_)),
            ],
        },
        {
            "id": "INC-20260418-001",
            "created_at": now - timedelta(seconds=3 * day),
            "location": "Gymnasium",
            "type": "Medical",
            "source": "MANUAL",
            "status": "RESOLVED",
            "severity": "Medium",
            "description": "Student injury during practice.",
            "reported_by": "school@demo.com",
            "timeline": [
                ("Manual report", None, now - timedelta(seconds=3 * day)),
                ("Resolved", None, now - timedelta(seconds=3 * day - 22 * min_)),
            ],
        },
    ]

    for s in seeds:
        inc = models.Incident(
            id=s["id"],
            school_id="default",
            created_at=s["created_at"],
            location=s["location"],
            type=s["type"],
            source=s["source"],
            status=s["status"],
            severity=s["severity"],
            probability=s.get("probability"),
            description=s.get("description"),
            reported_by=s.get("reported_by"),
        )
        db.add(inc)
        db.flush()
        for label, detail, ts in s["timeline"]:
            db.add(models.IncidentTimeline(
                incident_id=inc.id,
                timestamp=ts,
                label=label,
                detail=detail,
            ))
    db.commit()


@router.get("/schools/{school_id}/incidents", response_model=list[schemas.IncidentOut])
def get_incidents(school_id: str, db: Session = Depends(get_db)):
    incidents = (
        db.query(models.Incident)
        .filter_by(school_id=school_id)
        .order_by(models.Incident.created_at.desc())
        .all()
    )
    return [_incident_out(i) for i in incidents]


@router.post("/incidents", response_model=schemas.IncidentOut, status_code=201)
def create_incident(body: schemas.IncidentCreate, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    inc_id = _next_incident_id(db)
    inc = models.Incident(
        id=inc_id,
        school_id=body.school_id,
        created_at=now,
        location=body.location,
        type=body.type,
        source=body.source,
        status="NEW",
        severity=body.severity,
        probability=body.probability,
        description=body.description,
        reported_by=body.reported_by,
    )
    db.add(inc)
    db.flush()
    label = f"{body.source} detection" if body.source != "MANUAL" else "Manual report filed"
    detail = body.description or f"{body.reported_by} → {body.type} ({body.severity})"
    db.add(models.IncidentTimeline(
        incident_id=inc.id, timestamp=now, label=label, detail=detail,
    ))
    db.commit()
    db.refresh(inc)
    return _incident_out(inc)


@router.patch("/incidents/{incident_id}", response_model=schemas.IncidentOut)
def update_incident(
    incident_id: str,
    body: schemas.IncidentUpdate,
    db: Session = Depends(get_db),
):
    inc = db.query(models.Incident).filter_by(id=incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")

    now = datetime.now(timezone.utc)
    if body.status is not None and body.status != inc.status:
        inc.status = body.status
        db.add(models.IncidentTimeline(
            incident_id=inc.id, timestamp=now, label=f"Status → {body.status}",
        ))
    if body.audio_url is not None:
        inc.audio_url = body.audio_url
        db.add(models.IncidentTimeline(
            incident_id=inc.id, timestamp=now,
            label="Audio snippet attached", detail=body.audio_url,
        ))
    if body.video_url is not None:
        inc.video_url = body.video_url
        db.add(models.IncidentTimeline(
            incident_id=inc.id, timestamp=now,
            label="Video segment attached", detail=body.video_url,
        ))
    if body.video_confirmed is not None:
        inc.video_confirmed = body.video_confirmed
        if body.video_confirmed:
            db.add(models.IncidentTimeline(
                incident_id=inc.id, timestamp=now,
                label="Video AI confirmed", detail=f"Visual confirmation at {inc.location}",
            ))
    db.commit()
    db.refresh(inc)
    return _incident_out(inc)


@router.post("/incidents/{incident_id}/dispatch", response_model=schemas.IncidentOut)
def dispatch_unit(incident_id: str, db: Session = Depends(get_db)):
    inc = db.query(models.Incident).filter_by(id=incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    db.add(models.IncidentTimeline(
        incident_id=inc.id,
        timestamp=datetime.now(timezone.utc),
        label="Unit dispatched",
        detail="Patrol unit en route",
    ))
    db.commit()
    db.refresh(inc)
    return _incident_out(inc)
