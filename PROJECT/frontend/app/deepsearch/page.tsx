'use client'

import { useState } from 'react'
import { DeepSearchChat } from '../../components/DeepSearchChat'
import { Message } from '../../lib/types'

export default function DeepSearchPage() {
  const [messages, setMessages] = useState<Message[]>([])

  return (
    <main className="deepsearch-container max-w-3xl mx-auto  h-screen flex flex-col ">
      <DeepSearchChat messages={messages} setMessages={setMessages} />
    </main>
  )
}