'use client'

import { useState, useRef, useEffect } from "react"
import { Send, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "./ui/input"

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
  scrollToBottom()
  }, [messages])

  useEffect(() => {
  const container = chatContainerRef.current
  if (!container) return

  const handleScroll = () => {
  const { scrollTop, scrollHeight, clientHeight } = container
  const atBottom = scrollHeight - scrollTop - clientHeight <= 5 
  setShowScrollButton(!atBottom)
  }

  container.addEventListener("scroll", handleScroll)
  handleScroll() // Ensure button is correctly set on first render

  return () => container.removeEventListener("scroll", handleScroll)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:4200/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      const assistantMessage: Message = { role: 'assistant', content: data.answer }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      const [isLoading, setIsLoading] = useState(false)
    }
  }

  return (
    <div className="flex flex-col h-[600px] max-w-2xl mx-auto">
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-sm p-2 rounded-lg ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-500'}`}>
              {message.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      {showScrollButton && (
        <Button className="absolute bottom-20 right-4 rounded-full" size="icon" onClick={scrollToBottom}>
          <ChevronDown className="h-4 w-4" />
        </Button>
      )}
      <form onSubmit={handleSubmit} className="p-4 border-t flex">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-1 mr-2"
        />
        <Button type="submit" disabled={isLoading}>
          {isLoading ? 'Sending...' : <Send className="h-4 w-4" />}
        </Button>
      </form>
    </div>
  )
}

