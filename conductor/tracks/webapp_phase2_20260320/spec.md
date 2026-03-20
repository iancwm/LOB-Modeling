# Specification: LOB Modeling Webapp - Phase 2 Frontend

## Overview

This track implements the frontend user interface for the LOB Modeling Webapp, providing an interactive React-based interface for visualizing market making algorithms. The frontend connects to the Phase 1 backend via REST API and WebSocket for real-time simulation updates.

## Functional Requirements

### 1. Application Shell
- **FR1.1:** Create a React + TypeScript application with Bootstrap CSS styling
- **FR1.2:** Implement responsive navigation shell with header and main content area
- **FR1.3:** Display application title "LOB Modeling Webapp" and version information

### 2. Model Selector Component
- **FR2.1:** Fetch available models from `GET /models` endpoint
- **FR2.2:** Display models in a dropdown/selector component with display names
- **FR2.3:** Show model description when a model is selected
- **FR2.4:** Load model details via `GET /models/{model_id}` on selection

### 3. Parameter Controls Component
- **FR3.1:** Dynamically render input controls based on model parameter schema
- **FR3.2:** Support numeric inputs with min/max validation
- **FR3.3:** Display parameter descriptions as tooltips or help text
- **FR3.4:** Pre-populate controls with default parameter values
- **FR3.5:** Validate all parameters before simulation submission

### 4. Simulation Execution
- **FR4.1:** Provide "Run Simulation" button to trigger single simulation
- **FR4.2:** Display loading state during simulation execution
- **FR4.3:** Show error messages for failed simulations
- **FR4.4:** Display simulation results upon completion

### 5. Results Visualization
- **FR5.1:** Render time series data using Plotly.js or Chart.js
- **FR5.2:** Display key metrics from simulation results
- **FR5.3:** Support multiple visualization types (line charts, bar charts)
- **FR5.4:** Provide chart titles and axis labels based on visualization spec

### 6. API Integration
- **FR6.1:** Create API service module for REST endpoint calls
- **FR6.2:** Implement error handling for API failures
- **FR6.3:** Use TypeScript interfaces for API request/response types

## Non-Functional Requirements

### Performance
- **NFR1:** Initial page load < 3 seconds on broadband connection
- **NFR2:** UI remains responsive during simulation execution
- **NFR3:** Charts render within 500ms of receiving data

### Code Quality
- **NFR4:** All components written in TypeScript with strict type checking
- **NFR5:** Follow Google TypeScript Style Guide
- **NFR6:** Component code coverage > 80%

### User Experience
- **NFR7:** Consistent Bootstrap-based styling throughout
- **NFR8:** Clear visual feedback for loading and error states
- **NFR9:** Responsive layout supporting desktop and tablet viewports

## Acceptance Criteria

1. ✅ Frontend application builds and runs without errors
2. ✅ Model selector successfully loads and displays available models from backend
3. ✅ Parameter controls render correctly for Kyle Model (V_0, SIGMA_G, SIGMA_T, N)
4. ✅ Simulation can be executed and results displayed
5. ✅ All TypeScript code passes linting (gts or eslint)
6. ✅ All tests pass (`npm test` or `pytest` for frontend tests)
7. ✅ No console errors in browser developer tools

## Out of Scope

- WebSocket real-time streaming (deferred to Phase 3)
- Advanced chart customization and export
- User authentication or session persistence
- Mobile-optimized touch interface
- Dark mode or theme switching
- Historical simulation comparison

## Technical Stack

- **Framework:** React 18+ with TypeScript
- **Styling:** Bootstrap 5 CSS
- **Charts:** Plotly.js or Chart.js with React wrapper
- **HTTP Client:** Axios or Fetch API
- **Build Tool:** Vite or Create React App
- **Testing:** Jest + React Testing Library

## Dependencies

See `conductor/tech-stack.md` for backend dependencies. Frontend-specific dependencies:

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "typescript": "^5.0.0",
  "bootstrap": "^5.3.0",
  "react-bootstrap": "^2.8.0",
  "plotly.js": "^2.27.0",
  "react-plotly.js": "^2.6.0",
  "axios": "^1.6.0"
}
```
