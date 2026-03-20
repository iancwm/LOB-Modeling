"""WebSocket connection manager for session handling."""

import asyncio
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        self._active_connections: Dict[str, WebSocket] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            session_id: Session identifier.
            websocket: WebSocket connection.
        """
        await websocket.accept()
        self._active_connections[session_id] = websocket
        self._session_locks[session_id] = asyncio.Lock()

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            session_id: Session identifier.
        """
        self._active_connections.pop(session_id, None)
        self._session_locks.pop(session_id, None)

    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> None:
        """Send a message to a specific session.

        Args:
            session_id: Session identifier.
            message: Message to send.
        """
        websocket = self._active_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)

    async def broadcast(
        self, message: Dict[str, Any], exclude: Optional[Set[str]] = None
    ) -> None:
        """Broadcast a message to all connected sessions.

        Args:
            message: Message to broadcast.
            exclude: Set of session IDs to exclude.
        """
        exclude = exclude or set()
        for session_id, websocket in self._active_connections.items():
            if session_id not in exclude:
                await websocket.send_json(message)

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get a lock for thread-safe session operations.

        Args:
            session_id: Session identifier.

        Returns:
            Async lock for the session.
        """
        return self._session_locks.get(session_id, asyncio.Lock())

    @property
    def active_connections(self) -> Dict[str, WebSocket]:
        """Get all active connections.

        Returns:
            Dictionary of session ID to WebSocket.
        """
        return self._active_connections.copy()
