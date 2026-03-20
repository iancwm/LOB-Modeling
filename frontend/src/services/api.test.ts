import { describe, it, expect, vi, beforeEach } from 'vitest'
import { getModels, getModelDetails, runSimulation } from './api'
import axios from 'axios'

// Mock axios
vi.mock('axios', () => {
  const mockGet = vi.fn()
  const mockPost = vi.fn()
  return {
    default: {
      create: vi.fn(() => ({
        get: mockGet,
        post: mockPost,
      })),
    },
  }
})

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('getModels', () => {
    it('fetches models from /models endpoint', async () => {
      const mockModels = [
        { id: 'kyle', displayName: 'Kyle Model' },
      ]
      const mockGet = vi.fn().mockResolvedValue({ data: { models: mockModels } })
      const axios = await import('axios')
      axios.default.create.mockReturnValue({ get: mockGet })

      const result = await getModels()

      expect(mockGet).toHaveBeenCalledWith('/models')
      expect(result).toEqual(mockModels)
    })
  })

  describe('getModelDetails', () => {
    it('fetches model details from /models/:id endpoint', async () => {
      const mockModel = { id: 'kyle', displayName: 'Kyle Model' }
      const mockGet = vi.fn().mockResolvedValue({ data: mockModel })
      const axios = await import('axios')
      axios.default.create.mockReturnValue({ get: mockGet })

      const result = await getModelDetails('kyle')

      expect(mockGet).toHaveBeenCalledWith('/models/kyle')
      expect(result).toEqual(mockModel)
    })
  })

  describe('runSimulation', () => {
    it('posts simulation request to /models/:id/simulate', async () => {
      const mockResult = { time_series: {}, metrics: {} }
      const mockPost = vi.fn().mockResolvedValue({ data: { results: mockResult } })
      const axios = await import('axios')
      axios.default.create.mockReturnValue({ post: mockPost })

      const params = { V_0: 5.0, N: 50 }
      const result = await runSimulation('kyle', params)

      expect(mockPost).toHaveBeenCalledWith('/models/kyle/simulate', params)
      expect(result).toEqual(mockResult)
    })
  })
})
