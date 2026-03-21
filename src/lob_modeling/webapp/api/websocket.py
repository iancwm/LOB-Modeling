"""WebSocket router for real-time simulation updates."""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..modules import get_module
from ..session.manager import WebSocketManager
from ..session.store import InMemorySessionStore

router = APIRouter(tags=["websocket"])


# Global instances (injected from main.py)
_session_store: InMemorySessionStore | None = None
_ws_manager: WebSocketManager | None = None


def set_dependencies(
    session_store: InMemorySessionStore, ws_manager: WebSocketManager
) -> None:
    """Set global dependencies for the router.

    Args:
        session_store: Session store instance.
        ws_manager: WebSocket manager instance.
    """
    global _session_store, _ws_manager
    _session_store = session_store
    _ws_manager = ws_manager


def get_session_store() -> InMemorySessionStore:
    """Get the session store instance.

    Returns:
        Session store instance.

    Raises:
        RuntimeError: If session store is not initialized.
    """
    if _session_store is None:
        raise RuntimeError("Session store not initialized")
    return _session_store


def get_ws_manager() -> WebSocketManager:
    """Get the WebSocket manager instance.

    Returns:
        WebSocket manager instance.

    Raises:
        RuntimeError: If WebSocket manager is not initialized.
    """
    if _ws_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return _ws_manager


async def run_simulation_stream(
    session_id: str,
    model_id: str,
    params: Dict[str, Any],
) -> None:
    """Run simulation and stream results to WebSocket.

    Args:
        session_id: Session identifier.
        model_id: Model identifier.
        params: Simulation parameters.
    """
    session_store = get_session_store()
    ws_manager = get_ws_manager()

    try:
        module = get_module(model_id)

        # Send progress start
        await ws_manager.send_to_session(
            session_id,
            {
                "type": "simulation_progress",
                "payload": {
                    "progress": 0,
                    "status": "Starting simulation...",
                },
            },
        )

        # Run simulation
        result = module.simulate(params)

        # Send progress complete
        await ws_manager.send_to_session(
            session_id,
            {
                "type": "simulation_progress",
                "payload": {
                    "progress": 100,
                    "status": "Complete",
                },
            },
        )

        # Update session with result
        session_store.update_result(session_id, result.to_dict())

        # Send result to WebSocket
        await ws_manager.send_to_session(
            session_id,
            {
                "type": "simulation_result",
                "payload": {
                    "sessionId": session_id,
                    "results": result.to_dict(),
                },
            },
        )
    except Exception as e:
        await ws_manager.send_to_session(
            session_id,
            {
                "type": "error",
                "payload": {"message": f"Simulation failed: {str(e)}"},
            },
        )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """Websocket endpoint for real-time simulation updates.

    Args:
        websocket: WebSocket connection.
        session_id: Session identifier.
    """
    session_store = get_session_store()
    ws_manager = get_ws_manager()

    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            session = session_store.get(session_id)

            if not session:
                await websocket.send_json(
                    {
                        "type": "error",
                        "payload": {"message": "Session not found or expired"},
                    }
                )
                break

            if data.get("type") == "update_params":
                # Update session params and trigger re-simulation
                session.params.update(data.get("payload", {}))
                asyncio.create_task(
                    run_simulation_stream(
                        session_id,
                        session.model_id,
                        session.params,
                    )
                )

            elif data.get("type") == "stop_stream":
                break

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id)
        session_store.delete(session_id)
