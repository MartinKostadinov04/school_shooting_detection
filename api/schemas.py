"""
Pydantic schemas — mirror the TypeScript types in frontend/src/types/index.ts.
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


# ---------- Auth ----------

class LoginRequest(BaseModel):
    email: str
    password: str


class AuthUser(BaseModel):
    email: str
    role: Literal["school", "police"]
    displayName: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


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


class IncidentUpdate(BaseModel):
    status: Optional[IncidentStatus] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    video_confirmed: Optional[bool] = None


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
