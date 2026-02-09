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

## Usage

You can run the models using the `Makefile` commands:

```bash
make run-kyle
make run-almgren
make run-glosten
make run-criscuolo
```

Or by importing them in your Python scripts:

```python
from src.lob_modeling.models.kyle import KyleModel

model = KyleModel()
model.one_period_price()
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

### Criscuolo & Waehlbroeck
Attempts to replicate the results of Criscuolo & Waehlbroek (2014) on the effects of stochastic volatility on participation rate schedules.

## Testing

Run unit tests with:
```bash
make test
```
