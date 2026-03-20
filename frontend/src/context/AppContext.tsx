import React, { createContext, useContext, useState, ReactNode } from 'react'

export interface ModelParameter {
  name: string
  type: string
  default: number
  min?: number
  max?: number
  description: string
}

export interface ModelInfo {
  id: string
  displayName: string
  description: string
  parameters: Record<string, ModelParameter>
}

export interface SimulationResult {
  time_series: Record<string, number[]>
  metrics: Record<string, number>
}

interface AppState {
  models: ModelInfo[]
  selectedModel: ModelInfo | null
  parameters: Record<string, number>
  results: SimulationResult | null
  loading: boolean
  error: string | null
}

interface AppContextType extends AppState {
  setModels: (models: ModelInfo[]) => void
  setSelectedModel: (model: ModelInfo | null) => void
  setParameters: (params: Record<string, number>) => void
  setResults: (results: SimulationResult | null) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>({
    models: [],
    selectedModel: null,
    parameters: {},
    results: null,
    loading: false,
    error: null,
  })

  const setModels = (models: ModelInfo[]) => {
    setState(prev => ({ ...prev, models }))
  }

  const setSelectedModel = (model: ModelInfo | null) => {
    setState(prev => ({ 
      ...prev, 
      selectedModel: model,
      parameters: model ? Object.fromEntries(
        Object.entries(model.parameters).map(([key, param]) => [key, param.default])
      ) : {},
    }))
  }

  const setParameters = (params: Record<string, number>) => {
    setState(prev => ({ ...prev, parameters: params }))
  }

  const setResults = (results: SimulationResult | null) => {
    setState(prev => ({ ...prev, results }))
  }

  const setLoading = (loading: boolean) => {
    setState(prev => ({ ...prev, loading }))
  }

  const setError = (error: string | null) => {
    setState(prev => ({ ...prev, error }))
  }

  return (
    <AppContext.Provider value={{ 
      ...state, 
      setModels, 
      setSelectedModel, 
      setParameters, 
      setResults, 
      setLoading, 
      setError 
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider')
  }
  return context
}
