from datetime import datetime, timezone
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text
)
from sqlalchemy.orm import relationship
from api.database import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role         = Column(String, nullable=False)   # "school" | "police"
    display_name = Column(String, nullable=False)


class Device(Base):
    __tablename__ = "devices"

    id        = Column(String, primary_key=True)
    name      = Column(String, nullable=False)
    type      = Column(String, nullable=False)      # "camera" | "microphone"
    location  = Column(String, nullable=False)
    school_id = Column(String, nullable=False, default="default")
    status    = Column(String, nullable=False, default="online")
    x         = Column(Float, nullable=False, default=0)
    y         = Column(Float, nullable=False, default=0)
    last_event = Column(String, nullable=True)
    last_seen  = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    feed_url   = Column(String, nullable=True)


class Incident(Base):
    __tablename__ = "incidents"

    id              = Column(String, primary_key=True)
    school_id       = Column(String, nullable=False, default="default")
    created_at      = Column(DateTime(timezone=True), default=_now)
    location        = Column(String, nullable=False)
    type            = Column(String, nullable=False)
    source          = Column(String, nullable=False)   # AUDIO-AI | VIDEO-AI | MANUAL
    status          = Column(String, nullable=False, default="NEW")
    severity        = Column(String, nullable=False, default="High")
    probability     = Column(Float, nullable=True)
    description     = Column(Text, nullable=True)
    reported_by     = Column(String, nullable=True)
    audio_url       = Column(String, nullable=True)
    video_url       = Column(String, nullable=True)
    video_confirmed = Column(Boolean, default=False)

    timeline = relationship(
        "IncidentTimeline",
        back_populates="incident",
        cascade="all, delete-orphan",
        order_by="IncidentTimeline.timestamp",
    )
    messages = relationship(
        "Message",
        back_populates="incident",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )


class IncidentTimeline(Base):
    __tablename__ = "incident_timeline"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    timestamp   = Column(DateTime(timezone=True), default=_now)
    label       = Column(String, nullable=False)
    detail      = Column(String, nullable=True)

    incident = relationship("Incident", back_populates="timeline")


class Message(Base):
    __tablename__ = "messages"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    incident_id         = Column(String, ForeignKey("incidents.id"), nullable=False)
    timestamp           = Column(DateTime(timezone=True), default=_now)
    sender              = Column(String, nullable=False)   # "school" | "police" | "system"
    text                = Column(Text, nullable=True)
    incident_report_json = Column(Text, nullable=True)    # JSON string when present

    incident = relationship("Incident", back_populates="messages")
