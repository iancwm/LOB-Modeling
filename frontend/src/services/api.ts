import axios from 'axios'

const API_BASE_URL = ''

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

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function getModels(): Promise<ModelInfo[]> {
  const response = await api.get('/models')
  return response.data.models
}

export async function getModelDetails(modelId: string): Promise<ModelInfo> {
  const response = await api.get(`/models/${modelId}`)
  return response.data
}

export async function runSimulation(
  modelId: string,
  params: Record<string, number>
): Promise<SimulationResult> {
  const response = await api.post(`/models/${modelId}/simulate`, params)
  return response.data.results
}
