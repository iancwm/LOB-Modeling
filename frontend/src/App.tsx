import { Container, Navbar, Nav } from 'react-bootstrap'
import { AppProvider } from './context/AppContext'
import ModelSelector from './components/ModelSelector'
import ParameterControls from './components/ParameterControls'
import ResultsChart from './components/ResultsChart'

function App() {
  return (
    <AppProvider>
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand href="/">LOB Modeling Webapp</Navbar.Brand>
          <Navbar.Text className="text-muted ms-3">v0.1.0</Navbar.Text>
        </Container>
      </Navbar>
      <Container className="mt-4">
        <main>
          <div className="row">
            <div className="col-md-4">
              <ModelSelector />
              <ParameterControls />
            </div>
            <div className="col-md-8">
              <ResultsChart />
            </div>
          </div>
        </main>
      </Container>
    </AppProvider>
  )
}

export default App
