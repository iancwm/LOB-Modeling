# Specification: Webapp Model Integration - All Models

## Overview

This track integrates all existing LOB models into the webapp platform. Currently, only the Kyle Model is integrated. This track will add wrappers for Almgren-Chriss, Glosten-Milgrom, De Prado, Criscuolo & Waehlbroeck, and Asset Option models, making them accessible through the web interface with parameter controls and visualizations.

## Models to Integrate

1. **Almgren-Chriss (2000)** - Optimal execution with linear impact costs
2. **Glosten-Milgrom (1985)** - Specialist market bid-ask spread model
3. **De Prado et al. (2012)** - VPIN and optimal execution horizon
4. **Criscuolo & Waehlbroeck (2014)** - Stochastic volatility optimal execution
5. **Asset Option** - Asset or nothing option pricing

## Functional Requirements

### 1. Model Wrappers
- **FR1.1:** Create `ModelModule` wrapper for each model in `modules/wrappers/`
- **FR1.2:** Implement `parameters` property with correct types, ranges, and defaults
- **FR1.3:** Implement `visualizations` property with appropriate chart definitions
- **FR1.4:** Implement `simulate()` method that calls existing model code
- **FR1.5:** Implement `get_educational_content()` with theory and equations
- **FR1.6:** Register each model in `modules/__init__.py`

### 2. Parameter Definitions
- **FR2.1:** Define appropriate min/max ranges for each parameter
- **FR2.2:** Set sensible default values for each parameter
- **FR2.3:** Provide clear descriptions for each parameter
- **FR2.4:** Ensure parameter types match model expectations (int vs float)

### 3. Visualization Specifications
- **FR3.1:** Define appropriate chart types for each model (line, bar, multi-line)
- **FR3.2:** Configure axis labels and formatting for each chart
- **FR3.3:** Map simulation output to visualization data correctly
- **FR3.4:** Include all relevant time series in visualizations

### 4. Educational Content
- **FR4.1:** Provide learning objectives for each model
- **FR4.2:** Write background theory summaries
- **FR4.3:** Include key equations with descriptions
- **FR4.4:** Add interpretation guides for results

### 5. Backend API
- **FR5.1:** Verify REST API endpoints work with all models
- **FR5.2:** Verify WebSocket streaming works with all models
- **FR5.3:** Test model comparison with multiple models
- **FR5.4:** Handle model-specific errors gracefully

## Non-Functional Requirements

### Performance
- **NFR1:** All model simulations complete within 5 seconds
- **NFR2:** Model metadata loads within 500ms
- **NFR3:** Memory usage remains stable across multiple simulations

### Code Quality
- **NFR4:** All wrappers follow Google Python Style Guide
- **NFR5:** All wrappers have type hints
- **NFR6:** All wrappers have comprehensive docstrings
- **NFR7:** Test coverage > 80% for new wrapper code

### User Experience
- **NFR8:** Consistent parameter UI across all models
- **NFR9:** Clear error messages for invalid parameters
- **NFR10:** Helpful tooltips and descriptions for all parameters

## Acceptance Criteria

1. ✅ All 5 models appear in the model selector dropdown
2. ✅ Each model displays correct parameters with appropriate ranges
3. ✅ Each model runs simulation and displays results
4. ✅ Each model shows appropriate visualizations
5. ✅ Model comparison works with any combination of models
6. ✅ All models pass backend linting (flake8, black, isort)
7. ✅ All wrapper tests pass
8. ✅ Educational content displays for each model

## Out of Scope

- Modifying existing model implementation code
- Adding new models not already in the codebase
- Changing model algorithms or calculations
- Mobile optimization for model-specific views
- Advanced chart customization per model

## Technical Implementation

### Wrapper Pattern
Each model wrapper extends `ModelModule` abstract base class:

```python
class AlmgrenChrissModule(ModelModule):
    @property
    def model_id(self) -> str:
        return "almgren_chriss"
    
    @property
    def display_name(self) -> str:
        return "Almgren-Chriss (2000)"
    
    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "ALPHA": ParameterSpec(...),
            # ... more parameters
        }
    
    def simulate(self, params: Dict[str, Any]) -> SimulationResult:
        # Call existing model code
        # Return formatted results
```

### File Structure
```
src/lob_modeling/webapp/modules/wrappers/
├── kyle_wrapper.py (existing)
├── almgren_chriss_wrapper.py (new)
├── glosten_milgrom_wrapper.py (new)
├── de_prado_wrapper.py (new)
├── criscuolo_waehlbroeck_wrapper.py (new)
└── asset_option_wrapper.py (new)
```

## Dependencies

No new dependencies required. Uses existing:
- FastAPI for API endpoints
- NumPy/Pandas for data handling
- Plotly for visualizations
