from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db import get_db
from grader.db.models import User


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the current user from a Clerk session token.

    Phase-1 stub: in dev we accept `Authorization: Dev <clerk_id>` and look up or
    create the matching user. Production will verify Clerk JWTs against Clerk's JWKS.
    """
    if authorization is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing auth")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "dev" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad auth scheme")

    result = await db.execute(select(User).where(User.clerk_id == token))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(clerk_id=token, email=f"{token}@dev.local")
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user
