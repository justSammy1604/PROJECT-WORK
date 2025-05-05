'use client'

import { useState, FormEvent, useRef, useEffect } from 'react'
import { DeepSearchMessage } from './DeepSearchMessage'
import { ThinkingBox } from './ThinkingBox'
import { Message } from '../lib/types'

interface DeepSearchChatProps {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

const LoadingIndicator = () => (
  <div className="flex justify-start p-3">
    <div className="flex items-center space-x-1.5 bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg px-4 py-2 rounded-2xl shadow-md">
      <span className="sr-only">Loading...</span>
      <div className="h-2 w-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="h-2 w-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="h-2 w-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce"></div>
    </div>
  </div>
)

export function DeepSearchChat({ messages, setMessages }: DeepSearchChatProps) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const trimmedInput = input.trim()
    if (!trimmedInput || isLoading) return

    const userMessage: Message = { role: 'user', content: trimmedInput }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/deepsearch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmedInput }),
      })

      setIsLoading(false)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || `Server error ${response.status}`)
      }

      const data = await response.json()
      if (typeof data.response !== 'string' || typeof data.thinking !== 'string') {
        throw new Error('Invalid response format from server')
      }

      // Add thinking process as a message
      const thinkingMessage: Message = { role: 'thinking', content: data.thinking }
      // Add bot response as a message
      const botMessage: Message = { role: 'bot', content: data.response }
      setMessages((prev) => [...prev, thinkingMessage, botMessage])

    } catch (error: any) {
      setIsLoading(false)
      const errorMessage: Message = {
        role: 'bot',
        content: `Error: ${error.message || 'Unable to get response.'}`,
      }
      setMessages((prev) => [...prev, errorMessage])
    }
  }

  const isInputEmpty = input.trim().length === 0

  return (
    <div className="flex justify-center items-center h-full p-4 bg-transparent">
      <div className="flex flex-col h-full w-full max-w-4xl bg-white/80 dark:bg-gray-900/80 backdrop-blur-2xl rounded-2xl shadow-xl overflow-hidden border border-gray-200/50 dark:border-gray-700/50">
        {/* Message List Area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 scroll-smooth"
        >
          {messages.length === 0 && (
            <div className="flex justify-center items-center h-full">
              <p className="text-gray-500 dark:text-gray-400 text-center text-base font-sans px-6">
                Ask anything about financial markets.
              </p>
            </div>
          )}

          {messages.map((message, index) => {
            if (message.role === 'thinking') {
              return <ThinkingBox key={`thinking-${index}`} content={message.content} />
            } else {
              return <DeepSearchMessage key={`msg-${index}`} message={message} />
            }
          })}

          {isLoading && <LoadingIndicator />}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200/40 dark:border-gray-700/40 p-3 md:p-4 bg-white/70 dark:bg-gray-900/70 backdrop-blur-md">
          <form onSubmit={handleSubmit} className="flex items-center space-x-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Send a message..."
              className="flex-1 px-4 py-2.5 rounded-xl border border-gray-300/70 dark:border-gray-600/70 bg-white/90 dark:bg-gray-800/90 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-gray-900 disabled:opacity-60 font-sans text-base transition-shadow duration-200 focus:shadow-md"
              disabled={isLoading}
              aria-label="Chat input"
            />
            <button
              type="submit"
              className="p-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-gray-900 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center aspect-square"
              disabled={isLoading || isInputEmpty}
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M3.105 3.105a.75.75 0 01.954-.277l13.386 5.02a.75.75 0 010 1.304l-13.386 5.02a.75.75 0 01-1.231-.977L4.6 10 2.828 5.354a.75.75 0 01.277-.954z" />
                <path d="M4.242 10.182a.75.75 0 01.977 1.231l-1.775 4.735a.75.75 0 01-.954.277L2 15.422V4.578l1.001-1.002a.75.75 0 01.954.277l1.775 4.735a.75.75 0 01-.487.994z" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}