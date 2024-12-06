'use client'

import { useState, useEffect, useRef } from 'react'
import { useChat } from '@/hooks/useChat'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Sidebar } from '@/components/Sidebar'
import { Message } from '@/types/chat'
import { Menu } from 'lucide-react'

interface ChatProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

export function Chat({ sidebarOpen, setSidebarOpen }: ChatProps) {
  const { messages, input, setInput, handleSubmit, isLoading } = useChat()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex h-screen bg-red-500 dark:bg-red-900">
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}  // Properly pass onClose function
      />
      <div className="flex-1 flex flex-col">
        <div className="p-4 flex items-center">
          <Button variant="ghost" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu className="text-white" />
          </Button>
        </div>
        <ScrollArea className="flex-1 p-4 relative">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-white bg-opacity-50 dark:bg-gray-800 dark:bg-opacity-50 p-4 rounded-lg">
              <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Chatbot</h1>
            </div>
          </div>
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </ScrollArea>
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 dark:border-gray-800 flex justify-center">
          <div className="flex items-center rounded-full bg-white dark:bg-gray-800 overflow-hidden h-8 max-w-[50%] w-full">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 border-none focus:ring-0 h-full px-4 bg-transparent"
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading} className="rounded-full h-full px-4">
              {isLoading ? 'Sending...' : 'Send'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

function ChatMessage({ message }: { message: Message }) {
  return (
    <div className={`flex items-start mb-4 ${message.role === 'user' ? 'justify-end' : ''}`}>
      {message.role === 'assistant' && (
        <Avatar className="mr-2">
          <AvatarImage src="/assistant-avatar.png" alt="Assistant" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}
      <div
        className={`p-3 rounded-lg ${
          message.role === 'user'
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
        }`}
      >
        {message.content}
      </div>
      {message.role === 'user' && (
        <Avatar className="ml-2">
          <AvatarImage src="/user-avatar.png" alt="User" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}

