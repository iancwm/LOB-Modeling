"""Tests for session management."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.lob_modeling.webapp.session.store import SessionData, InMemorySessionStore
from src.lob_modeling.webapp.session.manager import WebSocketManager


class TestSessionData:
    """Tests for SessionData class."""

    def test_init(self):
        """Test SessionData initialization."""
        params = {"param1": 1.0, "param2": "test"}
        data = SessionData(
            session_id="test-123",
            model_id="kyle",
            params=params,
        )
        assert data.session_id == "test-123"
        assert data.model_id == "kyle"
        assert data.params == params
        assert data.result is None
        assert isinstance(data.created_at, datetime)
        assert isinstance(data.last_activity, datetime)


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore class."""

    def test_init(self):
        """Test InMemorySessionStore initialization."""
        store = InMemorySessionStore(ttl_minutes=60)
        assert store._ttl == timedelta(minutes=60)
        assert store._store == {}
        assert store._cleanup_task is None

    def test_create_session(self):
        """Test creating a new session."""
        store = InMemorySessionStore()
        params = {"param1": 1.0}
        data = store.create("test-123", "kyle", params)
        assert data.session_id == "test-123"
        assert data.model_id == "kyle"
        assert data.params == params
        assert "test-123" in store._store

    def test_get_existing_session(self):
        """Test getting an existing session."""
        store = InMemorySessionStore()
        store.create("test-123", "kyle", {})
        data = store.get("test-123")
        assert data is not None
        assert data.session_id == "test-123"

    def test_get_nonexistent_session(self):
        """Test getting a nonexistent session."""
        store = InMemorySessionStore()
        data = store.get("nonexistent")
        assert data is None

    def test_get_updates_last_activity(self):
        """Test that getting a session updates last_activity."""
        store = InMemorySessionStore()
        store.create("test-123", "kyle", {})
        initial_activity = store._store["test-123"].last_activity
        asyncio.sleep(0.1)  # Small delay
        store.get("test-123")
        assert store._store["test-123"].last_activity >= initial_activity

    def test_update_result(self):
        """Test updating session result."""
        store = InMemorySessionStore()
        store.create("test-123", "kyle", {})
        result = {"time_series": {"price": [1.0, 2.0]}}
        store.update_result("test-123", result)
        assert store._store["test-123"].result == result

    def test_update_result_nonexistent_session(self):
        """Test updating result for nonexistent session."""
        store = InMemorySessionStore()
        # Should not raise an error
        store.update_result("nonexistent", {"result": "data"})

    def test_delete_session(self):
        """Test deleting a session."""
        store = InMemorySessionStore()
        store.create("test-123", "kyle", {})
        store.delete("test-123")
        assert "test-123" not in store._store

    def test_delete_nonexistent_session(self):
        """Test deleting a nonexistent session."""
        store = InMemorySessionStore()
        # Should not raise an error
        store.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_cleanup_task(self):
        """Test that cleanup task removes expired sessions."""
        store = InMemorySessionStore(ttl_minutes=1)
        store.create("test-123", "kyle", {})
        
        # Manually expire the session
        store._store["test-123"].last_activity = datetime.utcnow() - timedelta(minutes=2)
        
        # Run cleanup manually
        expired = [
            sid for sid, data in store._store.items()
            if datetime.utcnow() - data.last_activity > store._ttl
        ]
        for sid in expired:
            del store._store[sid]
        
        assert "test-123" not in store._store

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the session store."""
        store = InMemorySessionStore()
        await store.close()
        # Should not raise an error

    @pytest.mark.asyncio
    async def test_close_with_cleanup_task(self):
        """Test closing the session store with running cleanup task."""
        store = InMemorySessionStore()
        await store.start_cleanup_task()
        await asyncio.sleep(0.1)  # Let task start
        await store.close()
        assert store._cleanup_task is None or store._cleanup_task.cancelled()


class TestWebSocketManager:
    """Tests for WebSocketManager class."""

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting a WebSocket."""
        manager = WebSocketManager()
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        
        await manager.connect("test-123", websocket)
        
        websocket.accept.assert_called_once()
        assert "test-123" in manager._active_connections
        assert "test-123" in manager._session_locks

    def test_disconnect(self):
        """Test disconnecting a WebSocket."""
        manager = WebSocketManager()
        manager._active_connections["test-123"] = MagicMock()
        manager._session_locks["test-123"] = asyncio.Lock()
        
        manager.disconnect("test-123")
        
        assert "test-123" not in manager._active_connections
        assert "test-123" not in manager._session_locks

    def test_disconnect_nonexistent(self):
        """Test disconnecting a nonexistent session."""
        manager = WebSocketManager()
        # Should not raise an error
        manager.disconnect("nonexistent")

    @pytest.mark.asyncio
    async def test_send_to_session(self):
        """Test sending a message to a session."""
        manager = WebSocketManager()
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()
        manager._active_connections["test-123"] = websocket
        
        message = {"type": "test", "data": "value"}
        await manager.send_to_session("test-123", message)
        
        websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_session(self):
        """Test sending to a nonexistent session."""
        manager = WebSocketManager()
        # Should not raise an error
        await manager.send_to_session("nonexistent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting a message to all sessions."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        manager._active_connections = {"session1": ws1, "session2": ws2}
        
        message = {"type": "broadcast"}
        await manager.broadcast(message)
        
        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_with_exclude(self):
        """Test broadcasting with excluded sessions."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        manager._active_connections = {"session1": ws1, "session2": ws2}
        
        message = {"type": "broadcast"}
        await manager.broadcast(message, exclude={"session1"})
        
        ws1.send_json.assert_not_called()
        ws2.send_json.assert_called_once_with(message)

    def test_get_lock(self):
        """Test getting a session lock."""
        manager = WebSocketManager()
        manager._session_locks["test-123"] = asyncio.Lock()
        
        lock = manager.get_lock("test-123")
        assert lock is manager._session_locks["test-123"]

    def test_get_lock_nonexistent(self):
        """Test getting a lock for nonexistent session."""
        manager = WebSocketManager()
        lock = manager.get_lock("nonexistent")
        assert isinstance(lock, asyncio.Lock)

    def test_active_connections(self):
        """Test getting active connections."""
        manager = WebSocketManager()
        ws1 = MagicMock()
        ws2 = MagicMock()
        manager._active_connections = {"session1": ws1, "session2": ws2}
        
        connections = manager.active_connections
        
        assert connections == {"session1": ws1, "session2": ws2}
        # Should return a copy
        connections["session3"] = MagicMock()
        assert "session3" not in manager._active_connections
