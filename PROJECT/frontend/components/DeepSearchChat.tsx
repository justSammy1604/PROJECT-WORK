'use client'

import { useState, FormEvent, useRef, useEffect } from 'react'
import { DeepSearchMessage } from './DeepSearchMessage' 
import { ThinkingBox } from './ThinkingBox'       
import { Message } from '../lib/types'             
import { useRouter } from 'next/navigation';

interface DeepSearchChatProps {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

const LoadingIndicator = () => (
  <div className="flex justify-start p-3">
    <div className="flex items-center space-x-1.5 bg-gray-700 px-4 py-2 rounded-2xl shadow-md">
      <span className="sr-only">Loading...</span>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
    </div>
  </div>
)

export function DeepSearchChat({ messages, setMessages }: DeepSearchChatProps) {
  const router = useRouter();
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleGoBack = () => {
    router.back();
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

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
      const response = await fetch('http://localhost:4200/deepsearch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmedInput, history: messages }),
      })

      setIsLoading(false)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: `Server error ${response.status}` }))
        throw new Error(errorData.message || `Server error ${response.status}`)
      }

      const data = await response.json()
      // Basic validation, adjust as needed for your actual API response structure
      if (typeof data.thinking !== 'string' || !Array.isArray(data.history)) {
        throw new Error('Invalid response format from server')
      }

      setMessages(data.history) // Assuming history contains all messages including bot response and thinking steps
    } catch (error: any) {
      setIsLoading(false)
      const errorMessage: Message = {
        role: 'bot', // Or 'assistant' if that's your type
        content: `Error: ${error.message || 'Unable to get response.'}`,
      }
      setMessages((prev) => [...prev, errorMessage])
    }
  }

  const isInputEmpty = input.trim().length === 0

  return (
    <div className="flex flex-col w-full max-w-3xl md:max-w-4xl xl:max-w-5xl h-screen bg-black text-gray-100 font-sans rounded-lg shadow-xl overflow-hidden">
      <button
        onClick={handleGoBack}
        aria-label="Go back"
        className="absolute top-3 left-3 z-10 p-2 bg-gray-700/50 hover:bg-gray-600/70 text-gray-300 hover:text-white rounded-full transition-colors duration-150 backdrop-blur-sm"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
      </button>
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 scroll-smooth" 
      >
        {messages.length === 0 && !isLoading && (
          <div className="flex justify-center items-center h-full">
            <div className="text-center">
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
          if (message.role === 'thinking') {
            return <ThinkingBox key={`thinking-${index}`} content={message.content} />
          } else {
            return <DeepSearchMessage key={`msg-${index}`} message={message} />
          }
        })}
        {isLoading && messages.length > 0 && messages[messages.length -1].role === 'user' && <LoadingIndicator />}
      </div>

      <div className="p-3 md:p-6 bg-black border-t border-gray-800">
        <form
          onSubmit={handleSubmit}
          className="flex items-end space-x-3 w-full" 
        >
          <textarea
            ref={inputRef as any}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !isLoading && !isInputEmpty) {
                e.preventDefault();
                handleSubmit(e as any);
              }
            }}
            placeholder="Send a message..."
            className="flex-1 p-3.5 rounded-xl border border-gray-700 bg-gray-800 text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[50px] max-h-[200px] leading-tight"
            disabled={isLoading}
            aria-label="Chat input"
            rows={1}
          />
          <button
            type="button"
            onClick={() => {
              setMessages([]);
              inputRef.current?.focus();
            }}
            className="p-3 bg-gray-700 text-gray-300 rounded-xl hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50 transition-colors duration-200 h-[50px] w-[50px] flex items-center justify-center"
            disabled={isLoading && messages.length === 0}
            aria-label="Clear chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
          </button>
          <button
            type="submit"
            className="p-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center aspect-square h-[50px] w-[50px]"
            disabled={isLoading || isInputEmpty}
            aria-label="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M3.105 3.105a.75.75 0 01.954-.277l13.386 5.02a.75.75 0 010 1.304l-13.386 5.02a.75.75 0 01-1.231-.977L4.6 10 2.828 5.354a.75.75 0 01.277-.954z" />
            </svg>
          </button>
        </form>
        <p className="text-xs text-gray-500 text-center mt-2 px-3">
          You bare sole responsibility for the data that you ask for and for what is retreived.
        </p>
      </div>
    </div>
  )
}