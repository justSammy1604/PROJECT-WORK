'use client'

import { useState } from 'react'
import { DeepSearchChat } from '../../components/DeepSearchChat'
import { Message } from '../../lib/types'

export default function DeepSearchPage() {
  const [messages, setMessages] = useState<Message[]>([])

  return (
    <main className="min-h-screen bg-black text-gray-100 flex flex-col items-center justify-center p-0 m-0">
      <DeepSearchChat messages={messages} setMessages={setMessages} />
    </main>
  )
}