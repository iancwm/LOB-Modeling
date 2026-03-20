import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useWebSocketConnection, getConnectionStatusText, getConnectionStatusColor } from './useWebSocket'
import { ReadyState } from 'react-use-websocket'

describe('useWebSocketConnection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('provides connection state helpers', () => {
    const { result } = renderHook(() => useWebSocketConnection())

    expect(result.current.isConnected).toBeDefined()
    expect(result.current.isConnecting).toBeDefined()
    expect(result.current.isClosed).toBeDefined()
    expect(result.current.sendMessage).toBeDefined()
    expect(result.current.sendJson).toBeDefined()
  })
})

describe('getConnectionStatusText', () => {
  it('returns correct text for each state', () => {
    expect(getConnectionStatusText(ReadyState.CONNECTING)).toBe('Connecting')
    expect(getConnectionStatusText(ReadyState.OPEN)).toBe('Connected')
    expect(getConnectionStatusText(ReadyState.CLOSING)).toBe('Disconnecting')
    expect(getConnectionStatusText(ReadyState.CLOSED)).toBe('Disconnected')
    expect(getConnectionStatusText(ReadyState.UNINSTANTIATED)).toBe('Uninitialized')
  })
})

describe('getConnectionStatusColor', () => {
  it('returns correct color for each state', () => {
    expect(getConnectionStatusColor(ReadyState.OPEN)).toBe('success')
    expect(getConnectionStatusColor(ReadyState.CONNECTING)).toBe('warning')
    expect(getConnectionStatusColor(ReadyState.CLOSED)).toBe('danger')
    expect(getConnectionStatusColor(ReadyState.CLOSING)).toBe('danger')
    expect(getConnectionStatusColor(ReadyState.UNINSTANTIATED)).toBe('secondary')
  })
})
