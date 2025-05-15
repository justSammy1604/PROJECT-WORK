// ./components/DeepSearchMessage.tsx
'use client'

import { marked } from 'marked'
import { Message } from '../lib/types' // Ensure Message type is imported if not globally available

interface DeepSearchMessageProps {
  message: Message
}

export function DeepSearchMessage({ message }: DeepSearchMessageProps) {
  const isUser = message.role === 'user'
  const isError = message.role === 'bot' && message.content?.toLowerCase().startsWith('error:')

  // Basic markdown parsing (ensure necessary extensions/options if needed)
  // Consider using a sanitizer here in production (like DOMPurify) if content isn't trusted
  const htmlContent = marked.parse(message.content || '', { breaks: true }) // Added breaks: true for newline handling

  const alignmentContainerClasses = isUser ? 'flex justify-end' : 'flex justify-start'

  // Unified base class and conditional classes for elegance
  const bubbleBaseClasses =
    'p-3 px-4 rounded-2xl shadow-md max-w-[85%] md:max-w-[75%] backdrop-blur-md border text-base font-sans leading-relaxed break-words' // Added px-4, text-base, leading-relaxed, break-words

  const userBubbleClasses = `${bubbleBaseClasses} bg-blue-500 text-white border-transparent`
  const botBubbleClasses = `${bubbleBaseClasses} bg-black dark:bg-gray-700/70 text-white dark:text-gray-100 border-none dark:border-gray-600/50`
  const errorBubbleClasses = `${bubbleBaseClasses} bg-red-100/80 dark:bg-red-900/60 text-red-800 dark:text-red-100 border-red-300/50 dark:border-red-700/50` // Adjusted error text color slightly

  const bubbleClasses = isUser
    ? userBubbleClasses
    : isError
    ? errorBubbleClasses
    : botBubbleClasses

  // Simpler prose classes, relying on bubble text color and dark:prose-invert
  // Added [&_a] for link styling and [&_p] for margin control
  const proseClasses = `prose prose-sm dark:prose-invert max-w-none 
    ${isUser ? '[&_a]:text-blue-200 hover:[&_a]:text-blue-100' : '[&_a]:text-blue-600 dark:[&_a]:text-blue-400 hover:[&_a]:text-blue-700 dark:hover:[&_a]:text-blue-300'} 
    [&_p]:my-1 first:[&_p]:mt-0 last:[&_p]:mb-0` // Ensure links are visible on user blue background and adjust paragraph margins

  return (
    <div className={alignmentContainerClasses}>
      <div className={bubbleClasses}>
        <div
          className={proseClasses}
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
      </div>
    </div>
  )
}