from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database import get_db
from api import models, schemas

router = APIRouter()

_SEED_DEVICES = [
    ("CAM-EN-01", "Entrance Cam",       "camera",      "Main Entrance",     "online",    33, 92),
    ("MIC-EN-01", "Entrance Mic",       "microphone",  "Main Entrance",     "online",    33, 80),
    ("CAM-HW-01", "Hallway Cam West",   "camera",      "1st Floor Hallway", "online",    22, 64),
    ("CAM-HW-02", "Hallway Cam East",   "camera",      "1st Floor Hallway", "warning",   65, 64),
    ("CAM-CF-01", "Cafeteria Cam",      "camera",      "Cafeteria",         "online",    50, 10),
    ("MIC-CF-01", "Cafeteria Mic",      "microphone",  "Cafeteria",         "online",    58, 14),
    ("CAM-GY-01", "Gym Cam Court",      "camera",      "Gymnasium",         "triggered", 28, 42),
    ("MIC-GY-01", "Gym Mic",            "microphone",  "Gymnasium",         "triggered", 35, 50),
    ("CAM-ST-01", "Stage Cam",          "camera",      "Stage",             "online",    33, 27),
    ("CAM-CR-01", "Classroom Cam N",    "camera",      "Classroom Wing",    "online",    73, 28),
    ("CAM-CR-02", "Classroom Cam S",    "camera",      "Classroom Wing",    "online",    73, 52),
    ("CAM-SL-01", "Science Lab Cam",    "camera",      "Science Lab",       "online",    53, 78),
    ("CAM-CL-01", "Computer Lab Cam",   "camera",      "Computer Lab",      "online",    84, 78),
    ("MIC-CL-01", "Computer Lab Mic",   "microphone",  "Computer Lab",      "online",    88, 78),
    ("CAM-OF-01", "Offices Cam",        "camera",      "Offices",           "online",    17, 82),
]


def seed_devices(db: Session) -> None:
    if db.query(models.Device).count() > 0:
        return
    now = datetime.now(timezone.utc)
    for did, name, dtype, location, status, x, y in _SEED_DEVICES:
        db.add(models.Device(
            id=did, name=name, type=dtype, location=location,
            school_id="default", status=status, x=x, y=y, last_seen=now,
        ))
    db.commit()


def _out(d: models.Device) -> schemas.DeviceOut:
    return schemas.DeviceOut(
        id=d.id,
        name=d.name,
        type=d.type,
        location=d.location,
        status=d.status,
        x=d.x,
        y=d.y,
        lastEvent=d.last_event,
        lastSeen=d.last_seen.isoformat() if d.last_seen else datetime.now(timezone.utc).isoformat(),
        feedUrl=d.feed_url,
    )


@router.get("/schools/{school_id}/devices", response_model=list[schemas.DeviceOut])
def get_devices(school_id: str, db: Session = Depends(get_db)):
    devices = db.query(models.Device).filter_by(school_id=school_id).all()
    return [_out(d) for d in devices]


@router.patch("/devices/{device_id}/status", response_model=schemas.DeviceOut)
def update_device_status(
    device_id: str,
    body: schemas.DeviceStatusUpdate,
    db: Session = Depends(get_db),
):
    device = db.query(models.Device).filter_by(id=device_id).first()
    if not device:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Device not found")
    device.status = body.status
    device.last_seen = datetime.now(timezone.utc)
    db.commit()
    db.refresh(device)
    return _out(device)
