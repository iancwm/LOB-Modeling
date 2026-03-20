import { useEffect, useState } from 'react'
import { ModelInfo } from '../services/api'
import { getModels } from '../services/api'

export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadModels() {
      try {
        setLoading(true)
        const data = await getModels()
        setModels(data)
        setError(null)
      } catch (err) {
        setError('Failed to load models')
        console.error('Error loading models:', err)
      } finally {
        setLoading(false)
      }
    }
    loadModels()
  }, [])

  return { models, loading, error }
}
