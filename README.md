# LOB Modeling

This repository compiles a collection of fundamental market making models and explorations.

## Structure

*   `src/lob_modeling/models/`: Contains the model implementations.
    *   `kyle.py`: Kyle Model (1985)
    *   `almgren_chriss.py`: Almgren-Chriss (2000) optimal execution
    *   `glosten_milgrom.py`: Glosten-Milgrom (1985)
    *   `de_prado.py`: De Prado et al. (2012)
    *   `criscuolo_waehlbroeck.py`: Criscuolo & Waehlbroeck (2014)
    *   `asset_option.py`: Asset or Nothing Option pricing
*   `data/`: Sample data files.
*   `tests/`: Unit tests.

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

*   numpy: Numerical computing
*   scipy: Scientific computing and optimization
*   matplotlib: Data visualization
*   scipy.optimize: Constrained optimization algorithms

## Usage

### Example Notebooks

The repository includes Jupyter notebooks demonstrating the models:

*   **almgren_chriss_example.ipynb**: Demonstrates the Almgren-Chriss optimal execution model with quadratic programming and dynamic programming solutions
*   **criscuolo_waehlbroeck_example.ipynb**: Demonstrates the Criscuolo & Waehlbroeck stochastic volatility model for optimal execution with visualization of execution schedules and participation rates

To run the notebooks:
```bash
jupyter notebook almgren_chriss_example.ipynb
jupyter notebook criscuolo_waehlbroeck_example.ipynb
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

model = KyleModel()
model.one_period_price()
```

For the Criscuolo & Waehlbroeck model:

```python
from lob_modeling.models.criscuolo_waehlbroeck import Criscuolo2014

model = Criscuolo2014(
    KAPPA=3,
    THETA=0.01,
    GAMMA=0.1,
    T=0.5,
    N=4,
    S_0=100
)

opt_result = model.optimal_execution()
```

## Models

### Kyle Model
Features single period and multiperiod versions of the discretized Kyle model. Computes params for determining agents order flow at each time period.

### Almgren-Chriss
Optimal execution models deviating from the seminal work of Almgren & Chriss (2000). Includes optimal execution with linear impact costs and stochastic optimal control.

### Glosten-Milgrom
Simplest version - given some order book at each time, computes the expected bid and ask.

### De Prado
Models, calculations, and data feed for exploring/verifying results of Easley, de Prado, O'Hara (2012).

### Criscuolo & Waehlbroeck (2014)

Implements the stochastic volatility optimal execution model from Criscuolo & Waehlbroeck (2014). The model captures realistic market conditions by incorporating:

*   **Stochastic Volatility**: Time-dependent variant of the Heston model with mean reversion
*   **Market Impact**: Both temporary (alpha) and permanent impact costs
*   **Constrained Optimization**: Uses scipy's SLSQP optimizer with sensible initial conditions and bounds

The execution schedule minimizes total cost while accounting for volatility dynamics. See `criscuolo_waehlbroeck_example.ipynb` for a complete walkthrough with visualizations.

## Testing

Run unit tests with:
```bash
make test
```
