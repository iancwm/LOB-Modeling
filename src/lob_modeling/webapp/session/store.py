"""In-memory session store with TTL-based expiration."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class SessionData:
    """Data container for a simulation session."""

    def __init__(self, session_id: str, model_id: str, params: Dict[str, Any]):
        """Initialize session data.

        Args:
            session_id: Unique session identifier.
            model_id: Model identifier being simulated.
            params: Simulation parameters.
        """
        self.session_id = session_id
        self.model_id = model_id
        self.params = params
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.result: Optional[Dict[str, Any]] = None


class InMemorySessionStore:
    """In-memory session store with TTL-based expiration."""

    def __init__(self, ttl_minutes: int = 30):
        """Initialize the session store.

        Args:
            ttl_minutes: Time-to-live for sessions in minutes.
        """
        self._store: Dict[str, SessionData] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self) -> None:
        """Start background task to clean expired sessions."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background loop to remove expired sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.utcnow()
            expired = [
                sid for sid, data in self._store.items()
                if now - data.last_activity > self._ttl
            ]
            for sid in expired:
                del self._store[sid]

    def create(
        self, session_id: str, model_id: str, params: Dict[str, Any]
    ) -> SessionData:
        """Create a new session.

        Args:
            session_id: Unique session identifier.
            model_id: Model identifier.
            params: Simulation parameters.

        Returns:
            Created session data.
        """
        data = SessionData(session_id, model_id, params)
        self._store[session_id] = data
        return data

    def get(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session data or None if not found.
        """
        data = self._store.get(session_id)
        if data:
            data.last_activity = datetime.utcnow()
        return data

    def update_result(self, session_id: str, result: Dict[str, Any]) -> None:
        """Update session with simulation result.

        Args:
            session_id: Session identifier.
            result: Simulation result.
        """
        if session_id in self._store:
            self._store[session_id].result = result
            self._store[session_id].last_activity = datetime.utcnow()

    def delete(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: Session identifier.
        """
        self._store.pop(session_id, None)

    async def close(self) -> None:
        """Close the session store and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
