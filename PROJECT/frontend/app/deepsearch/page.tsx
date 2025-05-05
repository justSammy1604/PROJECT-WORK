'use client'

import { useState } from 'react'
import { DeepSearchChat } from '../../components/DeepSearchChat'

export default function DeepSearchPage() {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'bot'; content: string }>>([])

  return (
    <main className="deepsearch-container max-w-3xl mx-auto p-4 h-screen flex flex-col">
      <DeepSearchChat messages={messages} setMessages={setMessages} />
    </main>
  )
}