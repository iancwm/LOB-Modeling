# LOB Modeling Webapp — Design Specification

**Date:** 2026-03-20  
**Author:** LOB-Modeling Team  
**Status:** Draft → Pending Review  

---

## 1. Executive Summary

This document specifies the design for a production-ready web application that visualizes advanced market making algorithms for educational purposes. The platform transforms the existing Python-based LOB-Modeling library into an interactive, web-based learning tool for students and academics studying market microstructure theory.

### 1.1 Product Vision

To provide a robust, interactive educational platform that helps students and academics understand market making algorithms through real-time visualization, parameter exploration, and model-specific insights.

### 1.2 Target Audience

- **Primary:** University students and academics studying market microstructure
- **Secondary:** Self-learners, quantitative finance educators, training programs

### 1.3 Core Goals

1. Enable interactive exploration of market making models with real-time parameter adjustments
2. Provide model-specific visualizations that communicate each algorithm's unique insights
3. Support both local development and cloud-hosted deployment scenarios
4. Maintain production-quality code standards (testing, CI/CD, error handling, logging)

---

## 2. Architecture Overview

### 2.1 Project Structure

```
LOB-Modeling/
├── src/
│   └── lob_modeling/          # Existing Python package
│       ├── models/            # Model implementations (Kyle, Almgren-Chriss, etc.)
│       ├── utils/
│       └── webapp/            # NEW: FastAPI backend
│           ├── __init__.py
│           ├── main.py        # FastAPI application entry point
│           ├── api/           # REST and WebSocket routers
│           ├── modules/       # Module registry and interface
│           │   ├── __init__.py
│           │   ├── base.py    # ModelModule abstract base class
│           │   ├── registry.py # Module discovery and registration
│           │   └── wrappers/  # Model-specific adapters
│           └── session/       # Session management
│               ├── __init__.py
│               ├── manager.py # Session lifecycle management
│               └── store.py   # In-memory or Redis-backed storage
├── frontend/                   # NEW: React application
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── App.tsx
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml         # Local development orchestration
├── Dockerfile                 # Multi-stage build (backend + frontend)
├── docs/
└── tests/
    ├── test_webapp/          # Backend tests
    └── frontend/             # Frontend tests
```

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              React Frontend (Docker Container)           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │   Module     │  │   Module     │  │   Module     │   │    │
│  │  │   Dashboard  │  │   Visualizer │  │   Controls   │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  │                                                            │    │
│  │  ┌────────────────────────────────────────────────────┐   │    │
│  │  │           D3.js Visualization Engine               │   │    │
│  │  └────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket (real-time updates)
                              │ REST API (model discovery, config)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         Server Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            FastAPI Backend (Docker Container)            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │   REST       │  │  WebSocket   │  │   Module     │   │    │
│  │  │   Router     │  │   Manager    │  │   Registry   │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  │                                                            │    │
│  │  ┌────────────────────────────────────────────────────┐   │    │
│  │  │              Model Module Interface                │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │   │    │
│  │  │  │ Kyle   │ │ Almgren│ │ Glosten│ │  ...   │      │   │    │
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘      │   │    │
│  │  └────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend** | React 18+ | Component-based architecture, strong ecosystem |
| **Visualization** | D3.js v7+ | Maximum flexibility for custom financial charts |
| **Backend** | FastAPI (Python 3.10+) | Native integration with existing models, async support |
| **Real-time** | WebSocket (via `websockets`) | Bidirectional streaming for slider interactions |
| **API** | REST + WebSocket | REST for discovery/config, WS for live updates |
| **Container** | Docker (multi-stage) | Consistent local and cloud deployment |
| **Testing** | pytest (backend), Jest + React Testing Library (frontend) | Full test coverage |
| **CI/CD** | GitHub Actions | Automated testing, building, deployment |

### 2.3 Deployment Model

**Hybrid Deployment Strategy:**

1. **Local Mode:** Single Docker Compose stack
   ```bash
   docker-compose up
   # Access: http://localhost:3000
   ```

2. **Cloud Mode:** Same Docker image deployed to:
   - AWS ECS/Fargate
   - Google Cloud Run
   - Heroku Container Registry
   - Any Kubernetes cluster

---

## 3. Module System Design

### 3.1 Module Architecture

Each market making model is implemented as a self-contained module with:

- **Model Interface:** Standardized Python class implementing the model logic
- **Visualization Spec:** D3.js chart definitions specific to the model
- **Educational Content:** Theory, equations, parameter descriptions, interpretation guide
- **Configuration Schema:** JSON schema defining valid parameters and ranges

### 3.2 Module Interface (Backend)

```python
class ModelModule(ABC):
    """Base interface for all model modules."""
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier (e.g., 'kyle', 'almgren-chriss')."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description for educational context."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        pass
    
    @property
    @abstractmethod
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        pass
    
    @abstractmethod
    def simulate(self, params: Dict[str, Any]) -> SimulationResult:
        """Run simulation with given parameters."""
        pass
    
    @abstractmethod
    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide."""
        pass
```

### 3.3 Module Registry

Modules are registered explicitly in a central registry. Each existing model class in `src/lob_modeling/models/` is wrapped by a corresponding `ModelModule` adapter in `src/lob_modeling/webapp/modules/wrappers/`:

```python
# Module registry location: src/lob_modeling/webapp/modules/registry.py

from .wrappers.kyle_wrapper import KyleModelModule
from .wrappers.almgren_chriss_wrapper import AlmgrenChrissModelModule
from .wrappers.glosten_milgrom_wrapper import GlostenMilgromModelModule
from .wrappers.de_prado_wrapper import DePradoModelModule
from .wrappers.criscuolo_waehlbroeck_wrapper import CriscuoloWaehlbroeckModelModule
from .wrappers.asset_option_wrapper import AssetOptionModelModule

MODEL_MODULES = {
    "kyle": KyleModelModule,
    "almgren_chriss": AlmgrenChrissModelModule,
    "glosten_milgrom": GlostenMilgromModelModule,
    "de_prado": DePradoModelModule,
    "criscuolo_waehlbroeck": CriscuoloWaehlbroeckModelModule,
    "asset_option": AssetOptionModelModule,
}

def get_module(model_id: str) -> ModelModule:
    """Get a module instance by ID."""
    if model_id not in MODEL_MODULES:
        raise ModuleNotFoundError(f"Model '{model_id}' not found")
    return MODEL_MODULES[model_id]()

def list_modules() -> List[str]:
    """List all registered module IDs."""
    return list(MODEL_MODULES.keys())
```

**Note on Naming:** Model IDs use underscores (`almgren_chriss`) to match existing Python module naming, but URLs accept both underscores and hyphens (FastAPI router normalizes hyphens to underscores).

### 3.4 Module Enable/Disable Configuration

Modules can be enabled/disabled via environment variable or config file:

```yaml
# config/modules.yaml
enabled_modules:
  - kyle
  - almgren-chriss
  - glosten-milgrom
# disabled_modules:
#   - de-prado
#   - criscuolo-waehlbroeck
#   - asset-option
```

---

### 3.5 Session Management

Sessions manage the lifecycle of WebSocket connections and simulation state.

#### Session Lifecycle

```
┌─────────────┐     create      ┌─────────────┐
│   Client    │ ──────────────► │   Session   │
│             │                 │   Manager   │
└─────────────┘                 └─────────────┘
       │                               │
       │  update_params                │
       │ ─────────────────────────────►│
       │                               │ spawn simulation
       │                               │
       │  simulation_result            ▼
       │ ◄───────────────────── ┌─────────────┐
       │                        │  Model      │
       │                        │  Module     │
       │                        └─────────────┘
       │
       │  stop_stream / timeout
       │ ─────────────────────────────►
                                       │ cleanup
                                       ▼
                                 ┌─────────────┐
                                 │   Session   │
                                 │   Store     │
                                 └─────────────┘
```

#### Session Store Implementation

**Local Mode (Default):** In-memory storage with TTL-based expiration

```python
# src/lob_modeling/webapp/session/store.py

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class SessionData:
    def __init__(self, session_id: str, model_id: str, params: Dict[str, Any]):
        self.session_id = session_id
        self.model_id = model_id
        self.params = params
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.result: Optional[Dict] = None

class InMemorySessionStore:
    def __init__(self, ttl_minutes: int = 30):
        self._store: Dict[str, SessionData] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._cleanup_task = None
    
    async def start_cleanup_task(self):
        """Start background task to clean expired sessions."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.utcnow()
            expired = [
                sid for sid, data in self._store.items()
                if now - data.last_activity > self._ttl
            ]
            for sid in expired:
                del self._store[sid]
    
    def create(self, session_id: str, model_id: str, params: Dict[str, Any]) -> SessionData:
        data = SessionData(session_id, model_id, params)
        self._store[session_id] = data
        return data
    
    def get(self, session_id: str) -> Optional[SessionData]:
        data = self._store.get(session_id)
        if data:
            data.last_activity = datetime.utcnow()
        return data
    
    def update_result(self, session_id: str, result: Dict):
        if session_id in self._store:
            self._store[session_id].result = result
            self._store[session_id].last_activity = datetime.utcnow()
    
    def delete(self, session_id: str):
        self._store.pop(session_id, None)
    
    async def close(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
```

**Cloud Mode (Optional):** Redis-backed storage for horizontal scaling

```python
# src/lob_modeling/webapp/session/store.py (alternative)

import redis.asyncio as redis
import json

class RedisSessionStore:
    def __init__(self, redis_url: str, ttl_minutes: int = 30):
        self._redis = redis.from_url(redis_url)
        self._ttl = ttl_minutes * 60  # Convert to seconds
    
    async def create(self, session_id: str, model_id: str, params: Dict[str, Any]):
        data = {
            "session_id": session_id,
            "model_id": model_id,
            "params": params,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "result": None
        }
        await self._redis.setex(
            f"session:{session_id}",
            self._ttl,
            json.dumps(data)
        )
        return data
    
    async def get(self, session_id: str) -> Optional[Dict]:
        data = await self._redis.get(f"session:{session_id}")
        if data:
            parsed = json.loads(data)
            # Update last_activity to extend TTL
            parsed["last_activity"] = datetime.utcnow().isoformat()
            await self._redis.setex(
                f"session:{session_id}",
                self._ttl,
                json.dumps(parsed)
            )
            return parsed
        return None
```

#### WebSocket Manager

```python
# src/lob_modeling/webapp/api/websocket_manager.py

from fastapi import WebSocket
from typing import Dict, Set
import asyncio

class WebSocketManager:
    def __init__(self):
        self._active_connections: Dict[str, WebSocket] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._active_connections[session_id] = websocket
        self._session_locks[session_id] = asyncio.Lock()
    
    def disconnect(self, session_id: str):
        self._active_connections.pop(session_id, None)
        self._session_locks.pop(session_id, None)
    
    async def send_to_session(self, session_id: str, message: Dict):
        """Send a message to a specific session."""
        websocket = self._active_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)
    
    async def broadcast(self, message: Dict, exclude: Set[str] = None):
        """Broadcast to all connected sessions."""
        exclude = exclude or set()
        for session_id, websocket in self._active_connections.items():
            if session_id not in exclude:
                await websocket.send_json(message)
    
    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get a lock for thread-safe session operations."""
        return self._session_locks.get(session_id, asyncio.Lock())
```

#### Session API Endpoints

**Create Session:**
```python
# POST /api/models/{model_id}/stream
# Returns WebSocket URL and session ID

@router.post("/models/{model_id}/stream")
async def create_stream(model_id: str, params: Dict[str, Any]):
    session_id = generate_session_id()
    session = await session_store.create(session_id, model_id, params)
    
    # Start simulation in background
    asyncio.create_task(run_simulation_stream(session_id, model_id, params))
    
    return {
        "session_id": session_id,
        "websocket_url": f"ws://localhost:8000/ws/{session_id}"
    }
```

**WebSocket Handler:**
```python
# WebSocket endpoint: /ws/{session_id}

@websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            session = await session_store.get(session_id)
            
            if not session:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "Session not found or expired"}
                })
                break
            
            if data["type"] == "update_params":
                # Update session params and trigger re-simulation
                session.params.update(data["payload"])
                await run_simulation_stream(session_id, session.model_id, session.params)
            
            elif data["type"] == "stop_stream":
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id)
        await session_store.delete(session_id)
```

#### Session Cleanup Strategy

1. **TTL-based expiration:** Sessions expire after 30 minutes of inactivity
2. **Explicit cleanup:** Client can send `stop_stream` to immediately clean up
3. **Disconnect handling:** WebSocket disconnect triggers session deletion
4. **Background cleanup:** Periodic task removes expired sessions (in-memory mode)
5. **Redis TTL:** Automatic expiration (Redis mode)

---

## 4. API Specification

### 4.1 REST Endpoints

#### `GET /api/models`

Returns list of available model modules.

**Response:**
```json
{
  "models": [
    {
      "id": "kyle",
      "displayName": "Kyle Model (1985)",
      "description": "Single dealer model with asymmetric information",
      "parameters": {...},
      "visualizations": [...]
    }
  ]
}
```

#### `GET /api/models/{model_id}`

Returns detailed metadata for a specific model.

#### `POST /api/models/{model_id}/simulate`

Runs a single simulation with provided parameters.

**Request:**
```json
{
  "parameters": {
    "sigma_v": 0.5,
    "sigma_u": 1.0,
    "n_periods": 10
  }
}
```

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "results": {
    "time_series": [...],
    "metrics": {...}
  }
}
```

#### `POST /api/models/{model_id}/stream`

Initiates a WebSocket streaming session for real-time updates.

**Response:**
```json
{
  "websocket_url": "ws://localhost:8000/ws/sim_abc123",
  "session_id": "sim_abc123"
}
```

#### `POST /api/export`

Exports current configuration and results as JSON.

#### `POST /api/import`

Imports a previously exported configuration.

### 4.2 WebSocket Protocol

#### Connection

Client connects to: `ws://<host>:<port>/ws/{session_id}`

#### Client → Server Messages

```json
{
  "type": "update_params",
  "payload": {
    "parameter_name": "new_value"
  }
}
```

```json
{
  "type": "stop_stream"
}
```

#### Server → Client Messages

```json
{
  "type": "simulation_result",
  "payload": {
    "timestamp": "2026-03-20T10:30:00Z",
    "results": {
      "time_series": [...],
      "metrics": {...}
    }
  }
}
```

```json
{
  "type": "error",
  "payload": {
    "message": "Invalid parameter value",
    "details": {...}
  }
}
```

---

## 5. Frontend Architecture

### 5.1 Component Structure

```
src/
├── components/
│   ├── Layout/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── Footer.tsx
│   ├── ModuleDashboard/
│   │   ├── ModuleSelector.tsx
│   │   ├── ModuleHeader.tsx
│   │   └── ModuleContent.tsx
│   ├── Visualization/
│   │   ├── ChartContainer.tsx
│   │   ├── TimeSeriesChart.tsx
│   │   ├── ParameterSensitivityChart.tsx
│   │   └── ModelSpecificChart.tsx  (per-model custom charts)
│   ├── Controls/
│   │   ├── ParameterSlider.tsx
│   │   ├── ParameterInput.tsx
│   │   └── RunButton.tsx
│   └── Educational/
│       ├── TheorySection.tsx
│       ├── EquationsDisplay.tsx
│       └── InterpretationGuide.tsx
├── hooks/
│   ├── useWebSocket.ts
│   ├── useModelModule.ts
│   └── useSimulation.ts
├── services/
│   ├── api.ts
│   └── websocket.ts
└── types/
    └── index.ts
```

### 5.2 Real-Time Slider Architecture

```
┌─────────────────┐     param change      ┌─────────────────┐
│  ParameterSlider│ ────────────────────► │  WebSocket Hook │
└─────────────────┘                       └─────────────────┘
                                                   │
                                                   │ update_params
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │   FastAPI WS    │
                                          │    Handler      │
                                          └─────────────────┘
                                                   │
                                                   │ simulate()
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │   Model Module  │
                                          └─────────────────┘
                                                   │
                                                   │ result
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │   WebSocket     │
                                          │   Broadcast     │
                                          └─────────────────┘
                                                   │
                                                   │ simulation_result
                                                   │
                                                   ▼
┌─────────────────┐                       ┌─────────────────┐
│   D3.js Chart   │ ◄──────────────────── │  WebSocket Hook │
│   (Re-render)   │     new data          └─────────────────┘
└─────────────────┘
```

### 5.3 D3.js Visualization Strategy

Each model defines its specific visualizations in a `VisualizationSpec`:

```python
# Example: Kyle Model visualization spec
{
  "model_id": "kyle",
  "visualizations": [
    {
      "id": "price_discovery",
      "title": "Price Discovery Over Time",
      "type": "multi_line",
      "data_mapping": {
        "x": "time",
        "y": ["true_value", "market_price", "informed_estimate"],
      },
      "axes": {
        "x": {"label": "Time Period", "format": "integer"},
        "y": {"label": "Price", "format": "currency"}
      },
      "annotations": [
        {"type": "reference_line", "y": "true_value", "label": "True Value"}
      ]
    },
    {
      "id": "order_flow",
      "title": "Order Flow Dynamics",
      "type": "stacked_bar",
      "data_mapping": {
        "x": "time",
        "y": ["informed_order", "noise_order", "dealer_inventory"]
      }
    }
  ]
}
```

Frontend renders charts dynamically based on spec:

```typescript
// Generic chart renderer
const ChartRenderer = ({ spec, data }: { spec: VisualizationSpec; data: any }) => {
  switch (spec.type) {
    case 'multi_line':
      return <MultiLineChart spec={spec} data={data} />;
    case 'stacked_bar':
      return <StackedBarChart spec={spec} data={data} />;
    // ... per-model custom charts
  }
};
```

### 5.4 Visualization Spec Schema (JSON)

Visualization specs are transmitted from backend to frontend as JSON. The schema:

```typescript
// TypeScript type definition (frontend/src/types/index.ts)

interface VisualizationSpec {
  id: string;                    // Unique identifier within model
  title: string;                 // Display title
  type: ChartType;               // Type of chart
  description?: string;          // Optional description/help text
  data_mapping: DataMapping;     // How to map simulation data to chart
  axes: AxisConfig;              // Axis configuration
  annotations?: Annotation[];    // Optional annotations (reference lines, etc.)
  styling?: ChartStyle;          // Optional styling overrides
}

type ChartType = 
  | 'multi_line'                 // Multiple line series
  | 'single_line'                // Single line series
  | 'stacked_bar'                // Stacked bar chart
  | 'grouped_bar'                // Grouped/clustered bar chart
  | 'scatter'                    // Scatter plot
  | 'area'                       // Area chart
  | 'heatmap'                    // Heatmap / 2D color grid
  | 'custom';                    // Model-specific custom visualization

interface DataMapping {
  x: string;                     // Field name for x-axis
  y: string | string[];          // Field name(s) for y-axis
  y_series_labels?: Record<string, string>;  // Optional labels for each y series
}

interface AxisConfig {
  x: AxisSpec;
  y: AxisSpec | AxisSpec[];      // Single or per-series
}

interface AxisSpec {
  label: string;                 // Axis label
  format: 'integer' | 'float' | 'currency' | 'percentage' | 'datetime';
  scale?: 'linear' | 'log' | 'time';
  domain?: [number, number];     // Optional fixed domain
}

interface Annotation {
  type: 'reference_line' | 'region' | 'label';
  value?: number | string;       // For reference_line: y-value or x-value
  y?: number;                    // For reference_line
  x?: number;                    // For vertical reference line
  label?: string;
  color?: string;
  stroke_dasharray?: string;     // e.g., "5,5" for dashed
}

interface ChartStyle {
  colors?: string[];             // Custom color palette
  stroke_width?: number;
  point_radius?: number;
  show_legend?: boolean;
  show_grid?: boolean;
}

// Example: Serialized spec sent to frontend
const exampleSpec: VisualizationSpec = {
  "id": "price_discovery",
  "title": "Price Discovery Over Time",
  "type": "multi_line",
  "description": "Shows convergence of market price to true value",
  "data_mapping": {
    "x": "time",
    "y": ["true_value", "market_price", "informed_estimate"],
    "y_series_labels": {
      "true_value": "True Value (V)",
      "market_price": "Market Price (P)",
      "informed_estimate": "Informed Trader Estimate"
    }
  },
  "axes": {
    "x": {"label": "Time Period", "format": "integer", "scale": "linear"},
    "y": {"label": "Price ($)", "format": "currency", "scale": "linear"}
  },
  "annotations": [
    {
      "type": "reference_line",
      "y": 100,
      "label": "True Value",
      "color": "#00ff88",
      "stroke_dasharray": "5,5"
    }
  ],
  "styling": {
    "colors": ["#00d9ff", "#00ff88", "#ff6b6b"],
    "stroke_width": 2,
    "point_radius": 3,
    "show_legend": true,
    "show_grid": true
  }
};
```

**Backend Serialization:**

```python
# src/lob_modeling/webapp/modules/base.py

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Literal

@dataclass
class AxisSpec:
    label: str
    format: Literal['integer', 'float', 'currency', 'percentage', 'datetime']
    scale: Optional[Literal['linear', 'log', 'time']] = None
    domain: Optional[tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class DataMapping:
    x: str
    y: str | List[str]
    y_series_labels: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class VisualizationSpec:
    id: str
    title: str
    type: Literal['multi_line', 'single_line', 'stacked_bar', 'grouped_bar', 
                  'scatter', 'area', 'heatmap', 'custom']
    description: Optional[str] = None
    data_mapping: Optional[DataMapping] = None
    axes: Optional[Dict[str, Any]] = None
    annotations: Optional[List[Dict[str, Any]]] = None
    styling: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict for frontend."""
        result = {"id": self.id, "title": self.title, "type": self.type}
        if self.description:
            result["description"] = self.description
        if self.data_mapping:
            result["data_mapping"] = self.data_mapping.to_dict()
        if self.axes:
            result["axes"] = self.axes
        if self.annotations:
            result["annotations"] = self.annotations
        if self.styling:
            result["styling"] = self.styling
        return result
```

---

## 6. Educational Content Structure

### 6.1 Content Sections (Per Model)

Each model module includes:

1. **Learning Objectives** (2-3 bullet points)
   - What students should understand after exploring this model

2. **Background Theory** (~200-300 words)
   - Intuition and economic reasoning
   - Key assumptions
   - Historical context

3. **Key Equations** (LaTeX-rendered)
   - Core mathematical formulation
   - Variable definitions

4. **Interactive Visualization**
   - Real-time parameter exploration
   - Model-specific charts

5. **Interpretation Guide** (~150-200 words)
   - How to read the visualizations
   - What parameter changes mean
   - Common insights and pitfalls

### 6.2 Example: Kyle Model Content

```markdown
## Kyle Model (1985)

### Learning Objectives
- Understand how asymmetric information affects price formation
- Observe how informed traders balance profit vs. information revelation
- See how market makers set prices based on order flow

### Background Theory
The Kyle model describes a market with asymmetric information...

### Key Equations
$$
p_n = p_{n-1} + \lambda (\alpha_n + u_n)
$$

Where:
- $p_n$: Market price at period n
- $\lambda$: Market depth (price impact)
- $\alpha_n$: Informed trader's order
- $u_n$: Noise trader order flow

### Interpretation Guide
The price discovery chart shows how the market price converges to the true value...
```

---

## 7. Export/Import Feature

### 7.1 Persistence Strategy

**MVP Decision:** File-based export/import only (no database). This is intentional for the educational use case:

- **Pros:** Simpler architecture, no database migrations, works in local-only mode, easy sharing via LMS/email
- **Cons:** No simulation history, no server-side save/load, no collaborative features

**Future Enhancement (Post-MVP):** If user feedback indicates demand for persistence:

```python
# Optional persistence layer (not in MVP)
# src/lob_modeling/webapp/persistence/models.py

from sqlalchemy import Column, String, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Simulation(Base):
    __tablename__ = "simulations"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Optional auth
    model_id = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=True)
    simulations = relationship("Simulation", back_populates="user")
```

For MVP, simulations exist only in memory during the session and are preserved via export.

### 7.2 Export Format

```json
{
  "version": "1.0",
  "exported_at": "2026-03-20T10:30:00Z",
  "model_id": "kyle",
  "parameters": {
    "sigma_v": 0.5,
    "sigma_u": 1.0,
    "n_periods": 10
  },
  "results": {
    "time_series": [...],
    "metrics": {...}
  },
  "visualization_state": {
    "active_charts": ["price_discovery", "order_flow"],
    "chart_settings": {...}
  }
}
```

### 7.2 Export Workflow

1. User clicks "Export" button
2. Frontend collects current state (parameters, results, visualization settings)
3. Browser downloads JSON file: `kyle-model-2026-03-20.json`
4. User can share file via email, LMS, or version control

### 7.3 Import Workflow

1. User clicks "Import" button
2. File picker opens, user selects JSON file
3. Frontend validates schema
4. Application state restored (parameters, results, visualizations)
5. User can re-run simulation or explore from imported state

---

## 8. Error Handling

### 8.1 Backend Error Categories

| Error Type | HTTP Status | WebSocket Message | User Message |
|-----------|-------------|-------------------|--------------|
| Invalid Parameters | 400 | `error: invalid_params` | "Parameter value out of range" |
| Model Execution Error | 500 | `error: simulation_failed` | "Simulation failed: {reason}" |
| WebSocket Timeout | N/A | `error: connection_lost` | "Connection lost. Reconnecting..." |
| Module Not Found | 404 | N/A | "Model not available" |

### 8.2 Frontend Error Boundaries

- **Global Error Boundary:** Catches unhandled React errors, shows fallback UI
- **Chart Error Boundary:** Individual charts fail gracefully without crashing app
- **WebSocket Error Handler:** Auto-reconnect with exponential backoff

### 8.3 Logging

Backend uses structured logging:

```python
import logging

logger = logging.getLogger("lob_modeling.webapp")

logger.info(
    "simulation_started",
    extra={
        "model_id": "kyle",
        "session_id": "sim_abc123",
        "parameters": {...}
    }
)
```

---

## 8.3 Logging

Backend uses structured logging:

```python
import logging

logger = logging.getLogger("lob_modeling.webapp")

logger.info(
    "simulation_started",
    extra={
        "model_id": "kyle",
        "session_id": "sim_abc123",
        "parameters": {...}
    }
)
```

---

## 9. Performance Optimization

### 9.1 Real-Time Performance Requirements

**Target:** <100ms latency from slider movement to chart update.

### 9.2 Optimization Strategies

#### Debouncing Slider Input

Client-side debouncing prevents excessive WebSocket messages:

```typescript
// frontend/src/hooks/useDebouncedParams.ts

import { useCallback, useEffect, useRef } from 'react';

export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const callbackRef = useRef(callback);
  
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  return useCallback((...args: Parameters<T>) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      callbackRef.current(...args);
    }, delay);
  }, [delay]) as T;
}

// Usage: 150ms debounce for slider updates
const debouncedUpdateParams = useDebouncedCallback(
  (paramName: string, value: number) => {
    sendWebSocketMessage('update_params', { [paramName]: value });
  },
  150
);
```

#### Simulation Caching

Cache recent simulation results to avoid re-computation:

```python
# src/lob_modeling/webapp/session/cache.py

from functools import lru_cache
from typing import Dict, Any, Tuple
import hashlib
import json

class SimulationCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size
    
    def _compute_hash(self, model_id: str, params: Dict[str, Any]) -> str:
        """Compute a hash for model+params combination."""
        key_str = json.dumps({"model_id": model_id, "params": params}, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, model_id: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Get cached result if exists."""
        key = self._compute_hash(model_id, params)
        return self._cache.get(key)
    
    def set(self, model_id: str, params: Dict[str, Any], result: Dict):
        """Cache a simulation result."""
        key = self._compute_hash(model_id, params)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
    
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()

# Global cache instance
simulation_cache = SimulationCache(max_size=1000)
```

#### Lazy Chart Rendering

Only render visible charts, use React.memo for memoization:

```typescript
// frontend/src/components/Visualization/ChartContainer.tsx

import React, { memo, useMemo } from 'react';

interface ChartContainerProps {
  spec: VisualizationSpec;
  data: SimulationData;
  isVisible: boolean;  // Only render if in viewport
}

export const ChartContainer = memo(({ spec, data, isVisible }: ChartContainerProps) => {
  // Skip expensive D3 rendering if chart is not visible
  if (!isVisible) {
    return <div className="chart-placeholder" />;
  }
  
  // Memoize data transformation
  const chartData = useMemo(() => transformDataForD3(spec, data), [spec, data]);
  
  return (
    <div className="chart-container">
      <D3Chart spec={spec} data={chartData} />
    </div>
  );
}, (prev, next) => {
  // Custom comparison: only re-render if spec or data changed
  return (
    prev.isVisible === next.isVisible &&
    prev.spec.id === next.spec.id &&
    prev.data.timestamp === next.data.timestamp
  );
});
```

#### WebAssembly for Heavy Computations (Future)

For complex models (multi-period Almgren-Chriss), consider WebAssembly:

```python
# Future optimization: compile performance-critical code to WASM
# Example: Rust implementation of simulation core

# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
numpy = "0.19"

# lib.rs
#[wasm_bindgen]
pub fn run_simulation(model_id: &str, params: JsValue) -> JsValue {
    // Rust implementation of simulation
    // 10-100x faster than pure Python for numerical computation
}
```

### 9.3 Frontend Performance

- **Code Splitting:** Lazy-load model modules via React.lazy + Suspense
- **Virtual Scrolling:** For long time series, render only visible data points
- **Web Workers:** Offload D3 data transformation to background thread
- **Request Animation Frame:** Sync chart updates with browser refresh rate

```typescript
// frontend/src/hooks/useSimulationStream.ts

import { useEffect, useRef } from 'react';

export function useSimulationStream(sessionId: string) {
  const animationFrameRef = useRef<number | null>(null);
  const latestDataRef = useRef<SimulationData | null>(null);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'simulation_result') {
        // Store latest data
        latestDataRef.current = message.payload;
        
        // Schedule chart update on next animation frame
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        animationFrameRef.current = requestAnimationFrame(() => {
          updateCharts(latestDataRef.current);
        });
      }
    };
    
    return () => {
      ws.close();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [sessionId]);
}
```

---

## 10. Security & Access Control

### 10.1 MVP: Open Access

**Decision:** No authentication for MVP. The tool is educational and intended for broad access.

### 10.2 Rate Limiting (Recommended)

Even without auth, implement basic rate limiting to prevent abuse:

```python
# src/lob_modeling/webapp/api/middleware.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self._rate_limit = requests_per_minute
        self._request_counts: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old entries (older than 1 minute)
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip]
            if now - t < 60
        ]
        
        # Check rate limit
        if len(self._request_counts[client_ip]) >= self._rate_limit:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."}
            )
        
        self._request_counts[client_ip].append(now)
        response = await call_next(request)
        return response

# Usage in main.py
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
```

### 10.3 CORS Configuration

Configure CORS for production deployment:

```python
# src/lob_modeling/webapp/main.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://lob-modeling.example.com",  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10.4 Future: Optional Authentication

For restricted deployments (e.g., university courses), add optional auth:

```python
# Future enhancement (not in MVP)

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Validate JWT token and return user ID."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token validation failed")

# Protected endpoint example
@app.post("/api/models/{model_id}/simulate")
async def simulate(
    model_id: str,
    params: Dict[str, Any],
    user_id: str = Depends(get_current_user)  # Requires auth
):
    ...
```

---

## 11. Testing Strategy

### 9.1 Backend Testing (pytest)

```python
# tests/webapp/test_model_modules.py

def test_kyle_module_interface():
    module = KyleModelModule()
    assert module.model_id == "kyle"
    assert module.display_name == "Kyle Model (1985)"
    assert len(module.parameters) > 0

def test_kyle_simulation():
    module = KyleModelModule()
    result = module.simulate({"sigma_v": 0.5, "sigma_u": 1.0, "n_periods": 10})
    assert result is not None
    assert "time_series" in result

@pytest.mark.asyncio
async def test_websocket_streaming():
    async with websockets.connect("ws://localhost:8000/ws/test_session") as ws:
        await ws.send(json.dumps({"type": "update_params", "payload": {...}}))
        response = await ws.recv()
        assert response["type"] == "simulation_result"
```

### 9.2 Frontend Testing (Jest + React Testing Library)

```typescript
// tests/components/ParameterSlider.test.tsx

test('slider updates parameter value on change', () => {
  const onParameterChange = jest.fn();
  render(
    <ParameterSlider
      paramSpec={{ name: 'sigma_v', min: 0, max: 1, default: 0.5 }}
      onParameterChange={onParameterChange}
    />
  );
  
  const slider = screen.getByRole('slider');
  fireEvent.change(slider, { target: { value: '0.75' } });
  
  expect(onParameterChange).toHaveBeenCalledWith('sigma_v', 0.75);
});

test('chart re-renders when simulation result updates', async () => {
  const { container } = render(<TimeSeriesChart data={initialData} />);
  const initialPoints = container.querySelectorAll('.data-point');
  
  rerender(<TimeSeriesChart data={updatedData} />);
  await waitFor(() => {
    const updatedPoints = container.querySelectorAll('.data-point');
    expect(updatedPoints.length).not.toBe(initialPoints.length);
  });
});
```

### 9.3 Integration Testing

```python
# tests/integration/test_full_workflow.py

def test_full_simulation_workflow():
    # 1. Get model list
    response = client.get("/api/models")
    assert response.status_code == 200
    
    # 2. Run simulation
    result = client.post("/api/models/kyle/simulate", json={...})
    simulation_id = result.json()["simulation_id"]
    
    # 3. Connect to WebSocket
    ws = websockets.connect(f"ws://localhost:8000/ws/{simulation_id}")
    
    # 4. Send parameter update
    ws.send(json.dumps({"type": "update_params", "payload": {...}}))
    
    # 5. Receive streaming result
    response = asyncio.get_event_loop().run_until_complete(ws.recv())
    assert response["type"] == "simulation_result"
```

---

## 10. CI/CD Pipeline

### 10.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=src/lob_modeling --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd frontend && npm ci
      - name: Run tests
        run: cd frontend && npm test -- --coverage
      - name: Build
        run: cd frontend && npm run build

  build-docker:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t lob-modeling-webapp:${{ github.sha }} .
      - name: Push to registry (on main only)
        if: github.ref == 'refs/heads/main'
        run: |
          # Push to Docker Hub, GHCR, or cloud registry
          ...

  deploy:
    needs: build-docker
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to cloud
        # Deployment steps for AWS/GCP/Heroku
        ...
```

---

## 11. Development Roadmap

### Phase 1: Foundation (Weeks 1-3)

- [ ] Set up project structure (backend + frontend)
- [ ] Implement module interface and registry
- [ ] Build Kyle model module as reference implementation
- [ ] Create basic React shell with module selector
- [ ] Implement WebSocket communication
- [ ] Docker Compose for local development

### Phase 2: Core Features (Weeks 4-6)

- [ ] Implement D3.js chart rendering engine
- [ ] Build Kyle-specific visualizations
- [ ] Add educational content sections
- [ ] Implement export/import functionality
- [ ] Add error handling and logging
- [ ] Write comprehensive tests

### Phase 3: Model Expansion (Weeks 7-9)

- [ ] Add Almgren-Chriss module
- [ ] Add Glosten-Milgrom module
- [ ] Refine module interface based on learnings
- [ ] Add model comparison feature (side-by-side view)

### Phase 4: Polish & Deployment (Weeks 10-12)

- [ ] Accessibility audit and fixes
- [ ] Performance optimization (chart rendering, WebSocket efficiency)
- [ ] CI/CD pipeline setup
- [ ] Cloud deployment (AWS/GCP/Heroku)
- [ ] Documentation and user guide
- [ ] Beta testing with students

---

## 12. Success Criteria

### Functional Requirements

- [ ] All 6 model modules implemented and functional
- [ ] Real-time slider updates with <100ms latency
- [ ] Export/import works reliably across browsers
- [ ] Docker image runs locally and deploys to cloud

### Quality Requirements

- [ ] Backend test coverage ≥ 80%
- [ ] Frontend test coverage ≥ 70%
- [ ] CI/CD pipeline passes on all PRs
- [ ] No critical accessibility issues (WCAG 2.1 AA)

### Educational Effectiveness

- [ ] Students can complete assigned exercises using the tool
- [ ] Instructors can share configurations via export/import
- [ ] User testing shows clear understanding of model behavior

---

## 13. Appendix

### 13.1 Glossary

| Term | Definition |
|------|------------|
| **LOB** | Limit Order Book |
| **Market Maker** | Trader providing liquidity by posting bid/ask quotes |
| **Asymmetric Information** | Situation where some traders have private information |
| **Price Impact** | Change in price caused by a trade |
| **Execution Schedule** | Sequence of trades over time |

### 13.2 References

1. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading". *Econometrica*.
2. Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions". *Journal of Risk*.
3. Glosten, L.R. & Milgrom, P.R. (1985). "Bid, Ask and Transaction Prices in a Specialist Market". *Journal of Financial Economics*.

### 13.3 Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend  # Multi-stage build
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
      - SESSION_TTL_MINUTES=30
      # Optional: Redis for cloud deployment
      # - REDIS_URL=redis://redis:6379
    volumes:
      - ./src/lob_modeling:/app/src/lob_modeling:ro
      - ./config:/app/config:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: production
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8000/api
      - REACT_APP_WS_URL=ws://localhost:8000/ws
    depends_on:
      - backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Optional: Redis for session storage (cloud deployment)
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis_data:/data
  #   healthcheck:
  #     test: ["CMD", "redis-cli", "ping"]
  #     interval: 10s
  #     timeout: 5s
  #     retries: 5

# Optional: Named volumes
# volumes:
#   redis_data:
```

### 13.4 Dockerfile (Multi-Stage)

```dockerfile
# Dockerfile

# ============================================
# Stage 1: Backend (Python/FastAPI)
# ============================================
FROM python:3.10-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/lob_modeling ./src/lob_modeling
COPY config ./config

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.lob_modeling.webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 2: Frontend Build (Node.js)
# ============================================
FROM node:18-alpine as frontend-build

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ ./

# Build production bundle
ARG REACT_APP_API_URL=/api
ARG REACT_APP_WS_URL=ws://localhost/ws
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL

RUN npm run build

# ============================================
# Stage 3: Frontend Production (Nginx)
# ============================================
FROM nginx:alpine as frontend

# Copy custom nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from build stage
COPY --from=frontend-build /app/frontend/build /usr/share/nginx/html

# Health check endpoint
RUN echo "ok" > /usr/share/nginx/html/health

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]

# ============================================
# Stage 4: Development (optional)
# ============================================
FROM backend as development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Install nodenv for frontend development
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]  # Keep container running for dev
```

### 13.5 TypeScript Configuration

```json
// frontend/tsconfig.json

{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@components/*": ["components/*"],
      "@hooks/*": ["hooks/*"],
      "@services/*": ["services/*"],
      "@types/*": ["types/*"]
    }
  },
  "include": ["src"],
  "exclude": ["node_modules", "build"]
}
```

### 13.6 Responsive Design Guidelines

- **Desktop (≥1024px):** Full layout with sidebar navigation, multi-column charts
- **Tablet (768px-1023px):** Collapsible sidebar, single-column charts, touch-friendly sliders
- **Mobile (<768px):** Hamburger menu, stacked layout, simplified controls (MVP may exclude mobile)

**MVP Decision:** Desktop-first, tablet-supported. Mobile optimization deferred to post-MVP.

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-20 | LOB-Modeling Team | Initial specification |
| 1.1 | 2026-03-20 | LOB-Modeling Team | Added: project structure, session management, visualization spec schema, persistence clarification, performance optimization, security/rate limiting, Docker Compose, TypeScript config |
