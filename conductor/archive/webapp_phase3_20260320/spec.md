# Specification: LOB Modeling Webapp - Phase 3 Enhancements

## Overview

This track implements advanced functionality enhancements for the LOB Modeling Webapp, focusing on real-time data streaming and multi-model comparison capabilities. These features will significantly improve the analytical capabilities of the platform.

## Functional Requirements

### 1. WebSocket Real-Time Streaming
- **FR1.1:** Establish WebSocket connection to backend on component mount
- **FR1.2:** Stream simulation progress updates in real-time
- **FR1.3:** Display live updating charts during simulation execution
- **FR1.4:** Handle WebSocket connection errors gracefully
- **FR1.5:** Implement reconnection logic for dropped connections
- **FR1.6:** Show connection status indicator in UI

### 2. Multi-Model Comparison
- **FR2.1:** Allow selection of multiple models for comparison
- **FR2.2:** Run simulations for all selected models with synchronized parameters
- **FR2.3:** Display comparison chart with overlaid results from all models
- **FR2.4:** Show comparison metrics table (final price, variance, etc.)
- **FR2.5:** Enable toggling individual model visibility on chart
- **FR2.6:** Support different color schemes for each model
- **FR2.7:** Add legend with model names and line styles

### 3. Enhanced UI Components
- **FR3.1:** Create model multi-select component with checkboxes
- **FR3.2:** Create comparison view toggle (single vs. multi-model)
- **FR3.3:** Add connection status badge in navbar
- **FR3.4:** Create comparison metrics summary card

### 4. Backend WebSocket Support
- **FR4.1:** Implement WebSocket endpoint for simulation streaming
- **FR4.2:** Stream partial results as simulation progresses
- **FR4.3:** Support multiple concurrent WebSocket connections
- **FR4.4:** Add session management for WebSocket connections

## Non-Functional Requirements

### Performance
- **NFR1:** WebSocket connection established within 500ms
- **NFR2:** Real-time updates rendered within 100ms of receiving data
- **NFR3:** Multi-model comparison with up to 4 models renders within 1 second
- **NFR4:** Memory usage remains stable during extended WebSocket sessions

### Code Quality
- **NFR5:** All new code follows Google TypeScript Style Guide
- **NFR6:** Component test coverage > 80%
- **NFR7:** WebSocket integration tests included

### User Experience
- **NFR8:** Clear visual feedback for connection status
- **NFR9:** Smooth animations for real-time chart updates
- **NFR10:** Intuitive model selection interface

## Acceptance Criteria

1. ✅ WebSocket connection establishes successfully on page load
2. ✅ Simulation results stream in real-time with live chart updates
3. ✅ Connection status indicator shows connected/disconnected/reconnecting states
4. ✅ User can select 2-4 models for comparison
5. ✅ Comparison chart displays all selected models with distinct colors
6. ✅ Comparison metrics table shows key metrics for all models
7. ✅ User can toggle individual model visibility on comparison chart
8. ✅ All new components have passing tests
9. ✅ No memory leaks during extended WebSocket sessions

## Out of Scope

- Historical simulation persistence (save/load sessions)
- Advanced chart export functionality
- Model parameter synchronization across comparisons
- Mobile-optimized touch interface for comparison view
- Dark mode or theme switching
- Advanced statistical analysis of model differences

## Technical Stack

### Frontend
- **WebSocket API:** Native browser WebSocket or `react-use-websocket`
- **Charts:** Plotly.js with streaming updates via `Plotly.extendTraces`
- **State Management:** Extend existing AppContext with WebSocket state
- **UI Components:** React Bootstrap with custom multi-select

### Backend
- **WebSocket Router:** Extend existing FastAPI WebSocket router
- **Streaming:** Use `WebSocket.send_json()` for real-time updates
- **Session Management:** Extend existing `InMemorySessionStore`

## Dependencies

Additional frontend dependencies:
```json
{
  "react-use-websocket": "^4.5.0"
}
```

Backend: No new dependencies (uses existing FastAPI WebSocket support)

## API Changes

### New WebSocket Message Types
```typescript
// Client -> Server
{ type: "start_simulation", modelId: string, params: object }
{ type: "stop_stream" }

// Server -> Client
{ type: "simulation_progress", progress: number, partialResults: object }
{ type: "simulation_complete", results: object }
{ type: "error", message: string }
```

### New REST Endpoints
```
POST /models/compare - Run comparison for multiple models
GET /models/compare/:sessionId - Get comparison results
```
