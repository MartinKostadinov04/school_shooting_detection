"""
Pydantic schemas — mirror the TypeScript types in frontend/src/types/index.ts.
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


# ---------- Devices ----------

DeviceType   = Literal["camera", "microphone"]
DeviceStatus = Literal["online", "warning", "triggered", "offline"]


class DeviceOut(BaseModel):
    id: str
    name: str
    type: DeviceType
    location: str
    status: DeviceStatus
    x: float
    y: float
    lastEvent: Optional[str] = None
    lastSeen: str
    feedUrl: Optional[str] = None

    class Config:
        from_attributes = True


class DeviceStatusUpdate(BaseModel):
    status: DeviceStatus


# ---------- Incidents ----------

IncidentType     = Literal["Gunshot", "Suspicious Activity", "Fire", "Medical", "Other"]
IncidentSource   = Literal["AUDIO-AI", "VIDEO-AI", "MANUAL"]
IncidentStatus   = Literal["NEW", "ACKNOWLEDGED", "RESOLVED"]
IncidentSeverity = Literal["Low", "Medium", "High", "Critical"]


class TimelineEntryOut(BaseModel):
    id: str
    timestamp: str
    label: str
    detail: Optional[str] = None

    class Config:
        from_attributes = True


class IncidentOut(BaseModel):
    id: str
    createdAt: str
    location: str
    type: IncidentType
    source: IncidentSource
    status: IncidentStatus
    severity: IncidentSeverity
    description: Optional[str] = None
    probability: Optional[float] = None
    audioUrl: Optional[str] = None
    videoUrl: Optional[str] = None
    videoConfirmed: Optional[bool] = None
    # Peak number of simultaneously visible guns from the video segment.
    # camelCase here mirrors the TypeScript Incident.gunCount field.
    gunCount: Optional[int] = None
    reportedBy: Optional[str] = None
    timeline: list[TimelineEntryOut] = []

    class Config:
        from_attributes = True


class IncidentCreate(BaseModel):
    school_id: str = "default"
    location: str
    type: IncidentType
    source: IncidentSource
    severity: IncidentSeverity
    probability: Optional[float] = None
    description: Optional[str] = None
    reported_by: Optional[str] = None
    # Optional on creation — VIDEO-AI detections may already know the count;
    # AUDIO-AI ingests have no count yet and will PATCH it in later.
    gun_count: Optional[int] = None


class IncidentUpdate(BaseModel):
    status: Optional[IncidentStatus] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    video_confirmed: Optional[bool] = None
    gun_count: Optional[int] = None


class VideoPathSubmit(BaseModel):
    video_path: str
    location: Optional[str] = None


# ---------- Messages ----------

class MessageOut(BaseModel):
    id: str
    timestamp: str
    sender: str
    text: Optional[str] = None
    incidentReport: Optional[dict] = None
    incidentId: Optional[str] = None

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    sender: Literal["school", "police", "system"]
    text: Optional[str] = None
    incidentReport: Optional[dict] = None


# ---------- Ably ----------

class AblyTokenResponse(BaseModel):
    token: str
    expires: int
