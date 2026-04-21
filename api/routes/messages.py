import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db
from api import models, schemas

router = APIRouter()


def _msg_out(m: models.Message) -> schemas.MessageOut:
    incident_report = None
    if m.incident_report_json:
        try:
            incident_report = json.loads(m.incident_report_json)
        except Exception:
            pass
    return schemas.MessageOut(
        id=str(m.id),
        timestamp=m.timestamp.isoformat() if m.timestamp else datetime.now(timezone.utc).isoformat(),
        sender=m.sender,
        text=m.text,
        incidentReport=incident_report,
        incidentId=m.incident_id,
    )


def seed_messages(db: Session) -> None:
    if db.query(models.Message).count() > 0:
        return
    # Attach demo messages to the Gymnasium incident
    inc = db.query(models.Incident).filter_by(id="INC-20260421-004").first()
    if not inc:
        return
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    min_ = 60
    seeds = [
        (None, "system", "Secure channel established. School ↔ Police.", now - timedelta(seconds=6 * min_)),
        ("INC-20260421-004", "school", "Hearing loud bangs near the gym. Pulling up cameras now.", now - timedelta(seconds=4 * min_)),
        ("INC-20260421-004", "police", "Acknowledged. Units en route. Keep students sheltered.", now - timedelta(seconds=3 * min_)),
    ]
    for inc_id, sender, text, ts in seeds:
        db.add(models.Message(
            incident_id=inc_id or "INC-20260421-004",
            timestamp=ts,
            sender=sender,
            text=text,
        ))
    db.commit()


@router.get("/incidents/{incident_id}/messages", response_model=list[schemas.MessageOut])
def get_messages(incident_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(models.Message)
        .filter_by(incident_id=incident_id)
        .order_by(models.Message.timestamp)
        .all()
    )
    return [_msg_out(m) for m in messages]


@router.post("/incidents/{incident_id}/messages", response_model=schemas.MessageOut, status_code=201)
def send_message(
    incident_id: str,
    body: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    if not db.query(models.Incident).filter_by(id=incident_id).first():
        raise HTTPException(status_code=404, detail="Incident not found")
    msg = models.Message(
        incident_id=incident_id,
        timestamp=datetime.now(timezone.utc),
        sender=body.sender,
        text=body.text,
        incident_report_json=json.dumps(body.incidentReport) if body.incidentReport else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return _msg_out(msg)
