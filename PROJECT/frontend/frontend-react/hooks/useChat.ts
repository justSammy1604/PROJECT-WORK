import { useState, useEffect, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'
import { Message } from '@/types/chat'

export function useChat() {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const newSocket = io('http://localhost:5000')
    setSocket(newSocket)

    return () => {
      newSocket.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!socket) return

    socket.on('message', (message: string) => {
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'assistant', content: message },
      ])
      setIsLoading(false)
    })

    return () => {
      socket.off('message')
    }
  }, [socket])

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (!socket || !input.trim()) return

      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'user', content: input },
      ])
      socket.emit('message', input)
      setInput('')
      setIsLoading(true)
    },
    [socket, input]
  )

  return { messages, input, setInput, handleSubmit, isLoading }
}

