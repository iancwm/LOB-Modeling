import { useState } from 'react'
import { Card, Form, Button, Spinner, Alert, ProgressBar } from 'react-bootstrap'
import { useApp } from '../context/AppContext'
import { runSimulation } from '../services/api'

export default function ParameterControls() {
  const { selectedModel, parameters, setParameters, setResults, setLoading, loading, setError } = useApp()
  const [localParams, setLocalParams] = useState<Record<string, number>>(parameters)
  const [progress, setProgress] = useState<number>(0)

  const handleParamChange = (name: string, value: string) => {
    const numValue = parseFloat(value)
    if (!isNaN(numValue)) {
      setLocalParams(prev => ({ ...prev, [name]: numValue }))
    }
  }

  const handleRunSimulation = async () => {
    if (!selectedModel) return

    try {
      setLoading(true)
      setError(null)
      setProgress(0)
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)
      
      const result = await runSimulation(selectedModel.id, localParams)
      
      clearInterval(progressInterval)
      setProgress(100)
      setResults(result)
      
      // Reset progress after a short delay
      setTimeout(() => {
        setProgress(0)
        setLoading(false)
      }, 500)
    } catch (err) {
      setError('Simulation failed. Please check parameters and try again.')
      console.error('Simulation error:', err)
      setLoading(false)
      setProgress(0)
    }
  }

  const handleSave = () => {
    setParameters(localParams)
  }

  if (!selectedModel) {
    return (
      <Card className="mb-3">
        <Card.Body>
          <Alert variant="info">Please select a model first.</Alert>
        </Card.Body>
      </Card>
    )
  }

  return (
    <Card className="mb-3">
      <Card.Header>Parameters</Card.Header>
      <Card.Body>
        {Object.entries(selectedModel.parameters).map(([key, param]) => (
          <Form.Group key={key} className="mb-3">
            <Form.Label>
              {param.name}
              {param.description && (
                <Form.Text className="text-muted d-block">
                  {param.description}
                </Form.Text>
              )}
            </Form.Label>
            <Form.Control
              type="number"
              step="0.01"
              value={localParams[key] ?? param.default}
              min={param.min}
              max={param.max}
              onChange={(e) => handleParamChange(key, e.target.value)}
            />
            <Form.Range
              min={param.min || 0}
              max={param.max || 100}
              step={(param.max || 100 - (param.min || 0)) / 100}
              value={localParams[key] ?? param.default}
              onChange={(e) => handleParamChange(key, e.target.value)}
            />
          </Form.Group>
        ))}
        {loading && (
          <div className="mb-3">
            <ProgressBar 
              animated 
              now={progress} 
              label={progress === 100 ? 'Complete!' : `Running... (${progress}%)`}
              variant={progress === 100 ? 'success' : 'primary'}
            />
          </div>
        )}
        <div className="d-grid gap-2">
          <Button 
            variant="primary" 
            onClick={handleRunSimulation}
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Running...
              </>
            ) : (
              'Run Simulation'
            )}
          </Button>
          <Button 
            variant="outline-secondary" 
            onClick={handleSave}
          >
            Apply Parameters
          </Button>
        </div>
      </Card.Body>
    </Card>
  )
}
