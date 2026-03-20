import { Card, Alert } from 'react-bootstrap'
import Plot from 'react-plotly.js'

interface ComparisonResult {
  modelId: string
  modelName: string
  time_series: Record<string, number[]>
  metrics: Record<string, number>
}

interface ComparisonChartProps {
  results: ComparisonResult[]
}

export default function ComparisonChart({ results }: ComparisonChartProps) {
  if (!results || results.length === 0) {
    return (
      <Card className="mb-3">
        <Card.Header>Model Comparison</Card.Header>
        <Card.Body>
          <Alert variant="info">
            Select multiple models and run simulations to compare results.
          </Alert>
        </Card.Body>
      </Card>
    )
  }

  // Collect all time series data
  const allTraces: Plotly.ScatterData[] = []
  const colors = ['#007bff', '#28a745', '#dc3545', '#fd7e14']

  results.forEach((result, index) => {
    const timeData = result.time_series.time || []
    const color = colors[index % colors.length]

    // Add traces for each metric
    Object.entries(result.time_series)
      .filter(([key]) => key !== 'time')
      .forEach(([key, values]) => {
        allTraces.push({
          x: timeData,
          y: values,
          name: `${result.modelName} - ${key.replace(/_/g, ' ')}`,
          type: 'scatter',
          mode: 'lines',
          line: { color, width: 2 },
        })
      })
  })

  // Collect all metrics for comparison table
  const allMetrics = results.map(r => ({
    modelId: r.modelId,
    modelName: r.modelName,
    metrics: r.metrics,
  }))

  return (
    <Card className="mb-3">
      <Card.Header>
        Model Comparison ({results.length} models)
      </Card.Header>
      <Card.Body>
        <Plot
          data={allTraces}
          layout={{
            width: undefined,
            height: 500,
            title: 'Model Comparison Results',
            xaxis: { title: 'Time Period' },
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

        {allMetrics.length > 0 && (
          <div className="mt-4">
            <h5>Metrics Comparison</h5>
            <div className="table-responsive">
              <table className="table table-bordered table-striped">
                <thead className="table-dark">
                  <tr>
                    <th>Metric</th>
                    {allMetrics.map(m => (
                      <th key={m.modelId}>{m.modelName}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(allMetrics[0]?.metrics || {}).map(metric => (
                    <tr key={metric}>
                      <td className="fw-bold">
                        {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </td>
                      {allMetrics.map(m => {
                        const value = m.metrics[metric]
                        const values = allMetrics.map(x => x.metrics[metric])
                        const isBest = value === Math.max(...values)
                        const isWorst = value === Math.min(...values)
                        return (
                          <td
                            key={m.modelId}
                            className={
                              isBest && allMetrics.length > 1
                                ? 'table-success fw-bold'
                                : isWorst && allMetrics.length > 1
                                ? 'table-warning'
                                : ''
                            }
                          >
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  )
}
