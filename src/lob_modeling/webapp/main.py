"""FastAPI application entry point for LOB Modeling Webapp."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import rest as rest_router
from .api import websocket as websocket_router
from .session.manager import WebSocketManager
from .session.store import InMemorySessionStore

# Global instances
session_store: InMemorySessionStore | None = None
ws_manager: WebSocketManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    global session_store, ws_manager

    # Startup
    session_store = InMemorySessionStore(ttl_minutes=30)
    ws_manager = WebSocketManager()

    # Initialize router dependencies
    rest_router.set_dependencies(session_store)
    websocket_router.set_dependencies(session_store, ws_manager)

    # Start session cleanup task
    await session_store.start_cleanup_task()

    yield

    # Shutdown
    if session_store:
        await session_store.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="LOB Modeling Webapp",
        description="Interactive visualization platform for market making algorithms",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Frontend development
            "http://localhost:8080",  # Alternative frontend port
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(rest_router.router)
    app.include_router(websocket_router.router)

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Health status.
        """
        return {"status": "healthy"}

    # Dependency injection for routers
    @app.get("/api/dependencies")
    async def get_dependencies() -> dict[str, bool]:
        """Get dependency status.

        Returns:
            Dictionary with dependency status.
        """
        return {
            "session_store": session_store is not None,
            "ws_manager": ws_manager is not None,
        }

    return app


# Create application instance
app = create_app()
