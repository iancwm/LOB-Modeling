import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import ConnectionStatus from './ConnectionStatus'

// Mock the health check fetch
global.fetch = vi.fn()

describe('ConnectionStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows connecting state initially', () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}))
    
    render(<ConnectionStatus />)
    
    expect(screen.getByText('Connecting')).toBeInTheDocument()
  })

  it('shows connected state when backend is healthy', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy' }),
    } as Response)

    render(<ConnectionStatus />)
    
    // Wait for health check to complete
    await vi.waitFor(() => {
      expect(screen.getByText('Backend Connected')).toBeInTheDocument()
    })
  })

  it('shows disconnected state when backend is unavailable', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network error'))

    render(<ConnectionStatus />)
    
    // Wait for health check to fail
    await vi.waitFor(() => {
      expect(screen.getByText('Backend Disconnected')).toBeInTheDocument()
    })
  })
})
