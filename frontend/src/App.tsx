import { Container, Navbar, Nav } from 'react-bootstrap'

function App() {
  return (
    <>
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand href="/">LOB Modeling Webapp</Navbar.Brand>
          <Navbar.Text className="text-muted ms-3">v0.1.0</Navbar.Text>
        </Container>
      </Navbar>
      <Container className="mt-4">
        <main>
          <h1>Welcome to LOB Modeling Webapp</h1>
          <p className="lead">
            Interactive visualization platform for market making algorithms.
          </p>
        </main>
      </Container>
    </>
  )
}

export default App
