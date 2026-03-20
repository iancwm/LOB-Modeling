# Track Specification: LOB Modeling Webapp - Phase 1 Foundation

## Goal
Implement the foundational infrastructure for the LOB Modeling Webapp, including the FastAPI backend structure, React frontend shell, module interface, and Docker Compose setup for local development.

## Requirements

### Backend (FastAPI)
- Create `src/lob_modeling/webapp/` package structure
- Implement `ModelModule` abstract base class in `modules/base.py`
- Create module registry in `modules/registry.py`
- Implement Kyle Model wrapper as reference implementation
- Set up session management with in-memory storage
- Create REST API endpoints for model discovery
- Create WebSocket endpoint for real-time updates

### Frontend (React)
- Create `frontend/` directory with React + TypeScript setup
- Implement basic app shell with navigation
- Create module selector component
- Set up WebSocket hook for real-time communication
- Create basic parameter controls component

### Infrastructure
- Create `Dockerfile` with multi-stage build
- Create `docker-compose.yml` for local development
- Create `Justfile` with development commands
- Create `pyproject.toml` for Python package management
- Create `frontend/package.json` with React dependencies
- Create `frontend/tsconfig.json` for TypeScript

### Testing
- Create basic test structure for backend
- Create basic test structure for frontend
- Ensure all tests pass

## Success Criteria
- `docker-compose up` starts both backend and frontend
- Backend serves REST API at `http://localhost:8000/api`
- Frontend serves React app at `http://localhost:3000`
- Kyle Model module is accessible via API
- WebSocket connection works for real-time updates
- All lint checks pass (flake8, black, isort)
- All tests pass

## Out of Scope (Phase 2+)
- D3.js visualizations
- Educational content sections
- Export/import functionality
- Additional model wrappers (Almgren-Chriss, Glosten-Milgrom, etc.)
- Model comparison feature
- Production deployment
