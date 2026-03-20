import { useState } from 'react'
import { SimulationResult } from '../services/api'
import { runSimulation } from '../services/api'

export function useSimulation() {
  const [results, setResults] = useState<SimulationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const executeSimulation = async (
    modelId: string,
    params: Record<string, number>
  ) => {
    try {
      setLoading(true)
      setError(null)
      const data = await runSimulation(modelId, params)
      setResults(data)
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Simulation failed'
      setError(errorMessage)
      console.error('Simulation error:', err)
      throw err
    } finally {
      setLoading(false)
    }
  }

  const clearResults = () => {
    setResults(null)
    setError(null)
  }

  return { results, loading, error, executeSimulation, clearResults }
}
