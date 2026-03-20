import { useState } from 'react'
import { Container, Navbar, Nav, Button, ButtonGroup, Alert } from 'react-bootstrap'
import { AppProvider, useApp } from './context/AppContext'
import ModelSelector from './components/ModelSelector'
import ModelMultiSelect from './components/ModelMultiSelect'
import ParameterControls from './components/ParameterControls'
import ResultsChart from './components/ResultsChart'
import ComparisonChart from './components/ComparisonChart'
import ConnectionStatus from './components/ConnectionStatus'

function AppContent() {
  const { models } = useApp()
  const [viewMode, setViewMode] = useState<'single' | 'comparison'>('single')
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([])

  const handleComparisonToggle = () => {
    if (viewMode === 'single') {
      setViewMode('comparison')
      // Pre-select current model if available
      if (selectedModelIds.length === 0) {
        setSelectedModelIds(['kyle'])
      }
    } else {
      setViewMode('single')
    }
  }

  return (
    <>
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand href="/">LOB Modeling Webapp</Navbar.Brand>
          <Navbar.Text className="text-muted ms-3">v0.3.0</Navbar.Text>
          <ConnectionStatus />
          <Nav className="ms-auto">
            <ButtonGroup>
              <Button
                variant={viewMode === 'single' ? 'primary' : 'outline-primary'}
                size="sm"
                onClick={() => setViewMode('single')}
              >
                Single Model
              </Button>
              <Button
                variant={viewMode === 'comparison' ? 'primary' : 'outline-primary'}
                size="sm"
                onClick={handleComparisonToggle}
              >
                Compare Models
              </Button>
            </ButtonGroup>
          </Nav>
        </Container>
      </Navbar>
      <Container className="mt-4">
        <main>
          {viewMode === 'single' ? (
            <div className="row">
              <div className="col-md-4">
                <ModelSelector />
                <ParameterControls />
              </div>
              <div className="col-md-8">
                <ResultsChart />
              </div>
            </div>
          ) : (
            <div className="row">
              <div className="col-md-4">
                <ModelMultiSelect
                  models={models}
                  selectedModelIds={selectedModelIds}
                  onSelectionChange={setSelectedModelIds}
                />
                <Alert variant="info" className="mb-3">
                  <small>
                    Select 2-4 models to compare. Run simulations for each selected model to see comparison results.
                  </small>
                </Alert>
              </div>
              <div className="col-md-8">
                <ComparisonChart results={[]} />
              </div>
            </div>
          )}
        </main>
      </Container>
    </>
  )
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  )
}

export default App
