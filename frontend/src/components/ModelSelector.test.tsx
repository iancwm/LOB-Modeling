import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { AppProvider } from '../context/AppContext'
import ModelSelector from './ModelSelector'
import * as api from '../services/api'

// Mock the API module
vi.mock('../services/api', () => ({
  getModels: vi.fn(),
}))

const mockModels = [
  {
    id: 'kyle',
    displayName: 'Kyle Model (1985)',
    description: 'Single dealer model with asymmetric information',
    parameters: {
      V_0: {
        name: 'V_0',
        type: 'float',
        default: 5.0,
        min: 0.0,
        max: 100.0,
        description: 'Initial security value',
      },
    },
  },
]

describe('ModelSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state initially', () => {
    vi.mocked(api.getModels).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    )

    render(
      <AppProvider>
        <ModelSelector />
      </AppProvider>
    )

    expect(screen.getByText(/loading models/i)).toBeInTheDocument()
  })

  it('displays models when loaded successfully', async () => {
    vi.mocked(api.getModels).mockResolvedValue(mockModels)

    render(
      <AppProvider>
        <ModelSelector />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('Model Selection')).toBeInTheDocument()
    })

    const select = screen.getByRole('combobox')
    expect(select).toBeInTheDocument()
    expect(screen.getByText('Kyle Model (1985)')).toBeInTheDocument()
  })

  it('displays model description when selected', async () => {
    vi.mocked(api.getModels).mockResolvedValue(mockModels)

    render(
      <AppProvider>
        <ModelSelector />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/Single dealer model/i)).toBeInTheDocument()
    })
  })

  it('shows error message when loading fails', async () => {
    vi.mocked(api.getModels).mockRejectedValue(new Error('Failed to load'))

    render(
      <AppProvider>
        <ModelSelector />
      </AppProvider>
    )

    await waitFor(() => {
      expect(
        screen.getByText(/Failed to load models/i)
      ).toBeInTheDocument()
    })
  })
})
