'use client'

import { useState } from 'react'
import { Chat } from '@/components/Chat'

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <Chat sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
  )
}

