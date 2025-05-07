'use client'

import { useState, FormEvent, useRef, useEffect } from 'react'
import { DeepSearchMessage } from './DeepSearchMessage'
import { ThinkingBox } from './ThinkingBox'
import { Message } from '../lib/types' // Assuming this path is correct

interface DeepSearchChatProps {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

const LoadingIndicator = () => (
  <div className="flex justify-start p-3">
    {/* Adjusted for dark mode and consistency */}
    <div className="flex items-center space-x-1.5 bg-gray-700 px-4 py-2 rounded-2xl shadow-md">
      <span className="sr-only">Loading...</span>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
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
        body: JSON.stringify({ query: trimmedInput, history: messages }),
      })

      setIsLoading(false)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || `Server error ${response.status}`)
      }

      const data = await response.json()
      if (
        typeof data.response !== 'string' ||
        typeof data.thinking !== 'string' ||
        !Array.isArray(data.history)
      ) {
        throw new Error('Invalid response format from server')
      }

      setMessages(data.history)
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
    // Main container: Full height, black background
    <div className="flex flex-col items-center h-screen bg-black text-gray-100 font-sans">
      {/* Chat Area Container: Mimics ChatGPT width and centering */}
      <div className="flex flex-col h-full w-full max-w-3xl md:max-w-4xl xl:max-w-5xl">
        {/* Message List Area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 scroll-smooth"
        >
          {messages.length === 0 && !isLoading && (
            <div className="flex justify-center items-center h-full">
              <div className="text-center">
                {/* You can add a logo or a more prominent welcome message here */}
                <h1 className="text-3xl font-semibold text-gray-300 mb-4">
                  Deepsearch
                </h1>
                <p className="text-gray-400 text-base">
                  Ask anything about financial markets.
                </p>
              </div>
            </div>
          )}

          {messages.map((message, index) => {
            // Assuming DeepSearchMessage and ThinkingBox are styled for dark mode
            // or you pass theme props to them.
            // For now, we'll ensure they are distinct enough.
            if (message.role === 'thinking') {
              return <ThinkingBox key={`thinking-${index}`} content={message.content} />
            } else {
              return <DeepSearchMessage key={`msg-${index}`} message={message} />
            }
          })}
          {isLoading && <LoadingIndicator />}
        </div>

        {/* Input Area: Sticks to bottom, consistent width */}
        <div className="p-3 md:p-6 bg-black"> {/* Changed from bg-gray-900/70 to bg-black for solid background */}
          <form
            onSubmit={handleSubmit}
            className="flex items-end space-x-3 max-w-3xl mx-auto" // Center input form like ChatGPT
          >
            <textarea // Changed input to textarea for multi-line support
              ref={inputRef as any} // Keep ref, adjust type if needed for textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { // Submit on Enter, new line on Shift+Enter
                if (e.key === 'Enter' && !e.shiftKey && !isLoading && !isInputEmpty) {
                  e.preventDefault();
                  handleSubmit(e as any); // handleSubmit expects FormEvent, type assertion
                }
              }}
              placeholder="Send a message..."
              className="flex-1 p-3.5 rounded-xl border border-gray-700 bg-gray-800 text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[50px] max-h-[200px] leading-tight" // Adjusted padding and min-height
              disabled={isLoading}
              aria-label="Chat input"
              rows={1} // Start with one row, auto-expands
            />
            <button
              type="button" // Explicitly type as button
              onClick={() => {
                setMessages([]);
                inputRef.current?.focus(); // Re-focus input after clearing
              }}
              className="p-3 bg-gray-700 text-gray-300 rounded-xl hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50 transition-colors duration-200"
              disabled={isLoading && messages.length === 0} // Disable only if loading and no messages
              aria-label="Clear chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /> {/* Simple Plus, or use a Trash Icon */}
                 {/* Example Trash Icon:
                 <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12.56 0c1.153 0 2.243.096 3.222.261m3.222.261L11 5.25M11 5.25v_5.25" /> */}
              </svg>
            </button>
            <button
              type="submit"
              className="p-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center aspect-square h-[50px] w-[50px]" // Fixed size for send button
              disabled={isLoading || isInputEmpty}
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M3.105 3.105a.75.75 0 01.954-.277l13.386 5.02a.75.75 0 010 1.304l-13.386 5.02a.75.75 0 01-1.231-.977L4.6 10 2.828 5.354a.75.75 0 01.277-.954z" />
                {/* Removed the second path for a cleaner send icon, or use a different one */}
              </svg>
            </button>
          </form>
           {/* Optional: Small text below input, similar to ChatGPT */}
          <p className="text-xs text-gray-500 text-center mt-2 px-3">
            You bare sole responsibility for the data that you ask for and for what is retreived.
          </p>
        </div>
      </div>
    </div>
  )
}