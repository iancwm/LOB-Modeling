import { useEffect, useState } from 'react'
import { Badge } from 'react-bootstrap'

export default function ConnectionStatus() {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting')

  useEffect(() => {
    // Check backend health via HTTP instead of WebSocket
    const checkHealth = async () => {
      try {
        const response = await fetch('/health')
        if (response.ok) {
          setStatus('connected')
        } else {
          setStatus('disconnected')
        }
      } catch {
        setStatus('disconnected')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return 'success'
      case 'connecting':
        return 'warning'
      case 'disconnected':
        return 'danger'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Backend Connected'
      case 'connecting':
        return 'Connecting'
      case 'disconnected':
        return 'Backend Disconnected'
    }
  }

  return (
    <Badge bg={getStatusColor()} className="ms-2">
      {getStatusText()}
    </Badge>
  )
}
