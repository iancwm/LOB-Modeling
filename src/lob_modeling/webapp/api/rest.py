"""REST API router for model endpoints."""

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..modules import get_module, list_modules
from ..modules.base import SimulationResult

router = APIRouter(prefix="/models", tags=["models"])


# Global instances (injected from main.py)
_session_store = None


def set_dependencies(session_store) -> None:
    """Set global dependencies for the router.

    Args:
        session_store: Session store instance.
    """
    global _session_store
    _session_store = session_store


def get_session_store():
    """Get the session store instance.

    Returns:
        Session store instance.

    Raises:
        RuntimeError: If session store is not initialized.
    """
    if _session_store is None:
        raise RuntimeError("Session store not initialized")
    return _session_store


@router.get("")
async def list_models() -> Dict[str, List[Dict[str, Any]]]:
    """List all available model modules.

    Returns:
        Dictionary with list of model information.
    """
    models = []
    for model_id in list_modules():
        try:
            module = get_module(model_id)
            models.append(
                {
                    "id": module.model_id,
                    "displayName": module.display_name,
                    "description": module.description,
                    "parameters": {
                        name: spec.to_dict() for name, spec in module.parameters.items()
                    },
                    "visualizations": [viz.to_dict() for viz in module.visualizations],
                }
            )
        except Exception as e:
            # Skip models that fail to load
            continue

    return {"models": models}


@router.get("/{model_id}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """Get detailed metadata for a specific model.

    Args:
        model_id: Model identifier.

    Returns:
        Model metadata.

    Raises:
        HTTPException: If model not found.
    """
    try:
        module = get_module(model_id)
        return {
            "id": module.model_id,
            "displayName": module.display_name,
            "description": module.description,
            "parameters": {
                name: spec.to_dict() for name, spec in module.parameters.items()
            },
            "visualizations": [viz.to_dict() for viz in module.visualizations],
            "educationalContent": module.get_educational_content().to_dict(),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.post("/{model_id}/simulate")
async def simulate(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simulation with provided parameters.

    Args:
        model_id: Model identifier.
        params: Simulation parameters.

    Returns:
        Simulation result.

    Raises:
        HTTPException: If model not found or simulation fails.
    """
    try:
        module = get_module(model_id)
        result = module.simulate(params)
        return {
            "simulationId": str(uuid.uuid4()),
            "results": result.to_dict(),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/{model_id}/stream")
async def create_stream(model_id: str, params: Dict[str, Any]) -> Dict[str, str]:
    """Initiate a WebSocket streaming session for real-time updates.

    Args:
        model_id: Model identifier.
        params: Simulation parameters.

    Returns:
        WebSocket URL and session ID.

    Raises:
        HTTPException: If model not found.
    """
    session_store = get_session_store()

    try:
        # Verify model exists
        module = get_module(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Create session
    session_id = str(uuid.uuid4())
    session_store.create(session_id, model_id, params)

    return {
        "sessionId": session_id,
        "websocketUrl": f"/ws/{session_id}",
    }
