import { Card, Alert } from 'react-bootstrap'
import Plot from 'react-plotly.js'
import { useApp } from '../context/AppContext'

export default function ResultsChart() {
  const { results, selectedModel } = useApp()

  if (!results) {
    return (
      <Card className="mb-3">
        <Card.Header>Results</Card.Header>
        <Card.Body>
          <Alert variant="info">
            Run a simulation to see results.
          </Alert>
        </Card.Body>
      </Card>
    )
  }

  // Extract time series data for plotting
  const timeData = results.time_series.time || results.time_series.bucket || []
  const xLabel = results.time_series.time ? 'Time' : 'Bucket'
  const traces = Object.entries(results.time_series)
    .filter(([key]) => key !== 'time' && key !== 'bucket')
    .map(([key, values]) => ({
      x: timeData,
      y: values,
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: 'scatter' as const,
      mode: 'lines' as const,
    }))

  return (
    <Card className="mb-3">
      <Card.Header>
        Results: {selectedModel?.displayName}
      </Card.Header>
      <Card.Body>
        <Plot
          data={traces}
          layout={{
            width: undefined,
            height: 400,
            title: 'Simulation Results',
            xaxis: { title: xLabel + ' Period' },
            yaxis: { title: 'Value' },
            showlegend: true,
            legend: { x: 0, y: 1 },
          }}
          config={{
            responsive: true,
            displayModeBar: true,
          }}
          style={{ width: '100%' }}
        />
        
        {Object.keys(results.metrics).length > 0 && (
          <div className="mt-3">
            <h5>Key Metrics</h5>
            <div className="row">
              {Object.entries(results.metrics).map(([key, value]) => (
                <div key={key} className="col-md-4 mb-2">
                  <Card bg="light">
                    <Card.Body className="text-center">
                      <Card.Title style={{ fontSize: '0.9rem' }}>
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </Card.Title>
                      <Card.Text style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </Card.Text>
                    </Card.Body>
                  </Card>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  )
}
