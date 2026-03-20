import useWebSocket, { ReadyState } from 'react-use-websocket'

export interface WebSocketMessage {
  type: string
  payload?: any
}

export interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onError?: (error: Event) => void
  onOpen?: () => void
  onClose?: () => void
}

export function useWebSocketConnection(options: UseWebSocketOptions = {}) {
  const { onMessage, onError, onOpen, onClose } = options

  const { sendMessage, sendJson, readyState, lastMessage } = useWebSocket(
    getWebSocketUrl(),
    {
      share: true,
      retryOnError: true,
      shouldReconnect: () => true,
      reconnectInterval: 3000,
      maxReconnectInterval: 30000,
      reconnectAttempts: 10,
      onOpen: onOpen ? () => onOpen() : undefined,
      onClose: onClose ? () => onClose() : undefined,
      onError: onError,
      onMessage: (event) => {
        if (onMessage) {
          try {
            const message = JSON.parse(event.data)
            onMessage(message)
          } catch {
            console.error('Failed to parse WebSocket message:', event.data)
          }
        }
      },
    }
  )

  return {
    sendMessage,
    sendJson,
    readyState,
    lastMessage,
    isConnected: readyState === ReadyState.OPEN,
    isConnecting: readyState === ReadyState.CONNECTING,
    isClosed: readyState === ReadyState.CLOSED,
  }
}

function getWebSocketUrl(): string {
  // Use relative URL for WebSocket - Vite will proxy to backend
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host || 'localhost:3000'
  return `${wsProtocol}//${host}/ws`
}

export function getConnectionStatusText(readyState: ReadyState): string {
  switch (readyState) {
    case ReadyState.CONNECTING:
      return 'Connecting'
    case ReadyState.OPEN:
      return 'Connected'
    case ReadyState.CLOSING:
      return 'Disconnecting'
    case ReadyState.CLOSED:
      return 'Disconnected'
    case ReadyState.UNINSTANTIATED:
    default:
      return 'Uninitialized'
  }
}

export function getConnectionStatusColor(readyState: ReadyState): string {
  switch (readyState) {
    case ReadyState.OPEN:
      return 'success'
    case ReadyState.CONNECTING:
      return 'warning'
    case ReadyState.CLOSED:
    case ReadyState.CLOSING:
      return 'danger'
    default:
      return 'secondary'
  }
}
