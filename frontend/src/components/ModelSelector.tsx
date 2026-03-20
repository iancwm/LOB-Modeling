import { useEffect, useState } from 'react'
import { Form, Card, Spinner, Alert } from 'react-bootstrap'
import { useApp } from '../context/AppContext'
import { getModels } from '../services/api'

export default function ModelSelector() {
  const { models, selectedModel, setSelectedModel, setModels, setError } = useApp()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadModels() {
      try {
        setLoading(true)
        const data = await getModels()
        setModels(data)
        if (data.length > 0) {
          setSelectedModel(data[0])
        }
      } catch (err) {
        setError('Failed to load models. Ensure backend is running.')
        console.error('Error loading models:', err)
      } finally {
        setLoading(false)
      }
    }
    loadModels()
  }, [])

  if (loading) {
    return (
      <Card className="mb-3">
        <Card.Body>
          <Spinner animation="border" size="sm" /> Loading models...
        </Card.Body>
      </Card>
    )
  }

  return (
    <Card className="mb-3">
      <Card.Header>Model Selection</Card.Header>
      <Card.Body>
        <Form.Group>
          <Form.Label>Select Model</Form.Label>
          <Form.Select
            value={selectedModel?.id || ''}
            onChange={(e) => {
              const model = models.find(m => m.id === e.target.value)
              setSelectedModel(model || null)
            }}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.displayName}
              </option>
            ))}
          </Form.Select>
        </Form.Group>
        {selectedModel && (
          <Form.Text className="text-muted mt-2 d-block">
            {selectedModel.description}
          </Form.Text>
        )}
      </Card.Body>
    </Card>
  )
}
