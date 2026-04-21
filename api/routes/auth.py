import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from api.database import get_db
from api import models, schemas

router = APIRouter()

JWT_SECRET    = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_H  = 24


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _create_token(user: models.User) -> str:
    payload = {
        "sub": user.email,
        "role": user.role,
        "displayName": user.display_name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_H),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def seed_users(db: Session) -> None:
    demo = [
        ("school@demo.com", "school123", "school", "School Operator"),
        ("police@demo.com", "police123", "police", "Dispatch Officer"),
    ]
    for email, password, role, display_name in demo:
        if not db.query(models.User).filter_by(email=email).first():
            db.add(models.User(
                email=email,
                password_hash=_hash(password),
                role=role,
                display_name=display_name,
            ))
    db.commit()


@router.post("/auth/login", response_model=schemas.TokenResponse)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=body.email.lower().strip()).first()
    if not user or not _verify(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid email or password")
    token = _create_token(user)
    return schemas.TokenResponse(
        access_token=token,
        user=schemas.AuthUser(
            email=user.email,
            role=user.role,
            displayName=user.display_name,
        ),
    )


@router.post("/auth/logout", status_code=status.HTTP_200_OK)
def logout():
    return {"detail": "logged out"}


@router.get("/auth/me", response_model=schemas.AuthUser)
def me(authorization: str = Header(default="", alias="authorization")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    payload = decode_token(authorization.removeprefix("Bearer ").strip())
    return schemas.AuthUser(
        email=payload["sub"],
        role=payload["role"],
        displayName=payload["displayName"],
    )
