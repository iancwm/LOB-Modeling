# LOB Modeling

This repository compiles a collection of fundamental market making models and explorations. All models feature Google-style docstrings and type hints for improved usability and documentation.

## Structure

*   `src/lob_modeling/`: Main package directory.
    *   `models/`: Contains the model implementations.
        *   `kyle.py`: Kyle Model (1985) - Single dealer model with asymmetric information
        *   `almgren_chriss.py`: Almgren-Chriss (2000) - Optimal execution with linear impact costs
        *   `glosten_milgrom.py`: Glosten-Milgrom (1985) - Specialist market bid-ask spread model
        *   `de_prado.py`: De Prado et al. (2012) - VPIN and optimal execution horizon
        *   `criscuolo_waehlbroeck.py`: Criscuolo & Waehlbroeck (2014) - Stochastic volatility optimal execution
        *   `asset_option.py`: Asset or Nothing Option pricing
    *   `webapp/`: FastAPI-based web application for interactive model visualization.
        *   `api/`: REST and WebSocket routers.
        *   `modules/`: Model module registry and wrappers.
        *   `session/`: Session management for WebSocket connections.
    *   `utils/`: Utility functions.
*   `data/`: Sample data files.
*   `tests/`: Unit tests.
    *   `webapp/`: Webapp-specific tests.
*   `frontend/`: React + TypeScript frontend (in development).

## Installation

1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    make install
    ```
    Or manually:
    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

**Core:**
*   numpy: Numerical computing
*   pandas: Data manipulation and analysis
*   scipy: Scientific computing and optimization
*   matplotlib: Data visualization
*   plotly: Interactive visualizations
*   scikit-learn: Machine learning utilities
*   statsmodels: Statistical modeling
*   yfinance: Market data fetching

**Webapp:**
*   fastapi: Modern web framework for building APIs
*   uvicorn: ASGI server for FastAPI
*   websockets: WebSocket support for real-time updates
*   httpx: Async HTTP client for testing

**Development:**
*   flake8: Code linting
*   black: Code formatting
*   isort: Import sorting
*   pydocstyle: Docstring style checking
*   pytest: Testing framework
*   pytest-asyncio: Async test support
*   pytest-cov: Coverage reporting

## Usage

### Example Notebooks

The repository includes Jupyter notebooks demonstrating the models:

*   **kyle_model_example.ipynb**: Kyle Model price discovery and order flow dynamics
*   **almgren_chriss_example.ipynb**: Almgren-Chriss optimal execution with quadratic and dynamic programming
*   **glosten_milgrom_example.ipynb**: Glosten-Milgrom bid-ask spread evolution
*   **criscuolo_waehlbroeck_example.ipynb**: Stochastic volatility optimal execution
*   **de_prado_example.ipynb**: VPIN calculation and market microstructure analysis
*   **asset_option_example.ipynb**: Asset or nothing option pricing

To run the notebooks:
```bash
jupyter notebook kyle_model_example.ipynb
jupyter notebook almgren_chriss_example.ipynb
```

### Command Line Usage

You can run the models using the `Makefile` commands:

```bash
make run-kyle
make run-almgren
make run-glosten
make run-criscuolo
```

### Python API

Import and use models directly in Python scripts:

```python
from lob_modeling.models.kyle import KyleModel

model = KyleModel(
    V_0=5.0,
    SIGMA_G=0.4,
    SIGMA_T=0.2,
    N=50
)
result = model.one_period_price()
```

For the Almgren-Chriss model:

```python
from lob_modeling.models.almgren_chriss import AlmgrenChriss2000

model = AlmgrenChriss2000(
    ALPHA=1.0,
    ETA=5e-6,
    GAMMA=5e-5,
    LAMBDA=0.00009,
    SIGMA=0.495,
    N=50,
    T=0.025,
    X=500
)

opt_sale, inventory, expected_shortfall, variance_shortfall = model.basic_almgren(plot=True)
```

### Web Application

The repository includes a FastAPI-based web application for interactive visualization of market making models.

**Starting the Backend:**

```bash
# From the project root
uvicorn src.lob_modeling.webapp.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

**API Endpoints:**

- `GET /health` - Health check endpoint
- `GET /models` - List all available models
- `GET /models/{model_id}` - Get model metadata and parameters
- `POST /models/{model_id}/simulate` - Run a single simulation
- `POST /models/{model_id}/stream` - Create a WebSocket streaming session
- `GET /api/dependencies` - Check dependency status

**Interactive API Documentation:**

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**WebSocket Streaming:**

The webapp supports real-time simulation updates via WebSocket:

```python
import websockets
import asyncio

async def stream_simulation():
    async with websockets.connect("ws://localhost:8000/ws/session-id") as ws:
        await ws.send({"type": "update_params", "payload": {"param": "value"}})
        response = await ws.recv()
        print(response)

asyncio.run(stream_simulation())
```

**Running Tests:**

```bash
# Run all tests including webapp tests
make test

# Run only webapp tests
pytest tests/webapp/ -v

# Run with coverage
pytest tests/ --cov=src/lob_modeling/webapp --cov-report=html
```

## Models

### Kyle Model (1985)
Features single period and multiperiod versions of the discretized Kyle model. Computes parameters for determining agent order flow at each time period. The model demonstrates how informed traders balance profit against information revelation.

### Almgren-Chriss (2000)
Optimal execution model deviating from the seminal work of Almgren & Chriss (2000). Includes both dynamic programming (Bellman equation) and quadratic programming solutions for optimal trade execution with linear impact costs.

### Glosten-Milgrom (1985)
Simplified specialist market model that uses Bayesian updating to compute expected bid and ask prices based on observed order flow. Demonstrates how market makers learn from trades.

### De Prado et al. (2012)
Implements the De Prado framework for optimal execution horizon, including:
- VPIN (Volume-synchronized Probability of Informed Trading)
- BVC (Bulk Volume Classification)
- LOBSTER data integration
- Autoregressive order imbalance modeling

### Criscuolo & Waehlbroeck (2014)
Implements the stochastic volatility optimal execution model. The model captures realistic market conditions by incorporating:

*   **Stochastic Volatility**: Time-dependent variant of the Heston model with mean reversion
*   **Market Impact**: Both temporary (alpha) and permanent impact costs
*   **Constrained Optimization**: Uses scipy's SLSQP optimizer

The execution schedule minimizes total cost while accounting for volatility dynamics. See `criscuolo_waehlbroeck_example.ipynb` for a complete walkthrough with visualizations.

### Asset or Nothing Option
Binomial tree pricing model for asset or nothing call options. Pays the asset value if the asset price exceeds the strike at expiry.

## Development

### Code Quality

This project follows Google Python style guidelines. All code includes Google-style docstrings and type hints.

```bash
# Run all lint checks
make lint

# Format code
make format

# Check docstrings
make check-docstrings
```

### Testing

Run all tests with pytest:
```bash
make test
```

Run tests with coverage:
```bash
pytest tests/ --cov=src/lob_modeling --cov-report=html
```

Run specific test modules:
```bash
# Run only model tests
pytest tests/test_models.py -v

# Run only webapp tests
pytest tests/webapp/ -v
```

### Running Models

```bash
# Run individual models
make run-kyle
make run-almgren
make run-glosten
make run-criscuolo
```
