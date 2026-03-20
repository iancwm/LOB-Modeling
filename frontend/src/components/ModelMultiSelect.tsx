import { Form, Card } from 'react-bootstrap'
import { ModelInfo } from '../services/api'

interface ModelMultiSelectProps {
  models: ModelInfo[]
  selectedModelIds: string[]
  onSelectionChange: (modelIds: string[]) => void
}

export default function ModelMultiSelect({
  models,
  selectedModelIds,
  onSelectionChange,
}: ModelMultiSelectProps) {
  const handleModelToggle = (modelId: string) => {
    if (selectedModelIds.includes(modelId)) {
      // Remove if already selected (but keep at least 1)
      if (selectedModelIds.length > 1) {
        onSelectionChange(selectedModelIds.filter(id => id !== modelId))
      }
    } else {
      // Add if not selected (max 4 models)
      if (selectedModelIds.length < 4) {
        onSelectionChange([...selectedModelIds, modelId])
      }
    }
  }

  return (
    <Card className="mb-3">
      <Card.Header>
        Model Comparison
        <Form.Text className="text-muted d-block">
          Select {selectedModelIds.length}/4 models to compare
        </Form.Text>
      </Card.Header>
      <Card.Body>
        <Form>
          {models.map((model) => (
            <Form.Check
              key={model.id}
              type="checkbox"
              id={`model-${model.id}`}
              label={model.displayName}
              checked={selectedModelIds.includes(model.id)}
              onChange={() => handleModelToggle(model.id)}
              disabled={
                !selectedModelIds.includes(model.id) &&
                selectedModelIds.length >= 4
              }
            />
          ))}
        </Form>
      </Card.Body>
    </Card>
  )
}
