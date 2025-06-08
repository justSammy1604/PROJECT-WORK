'use client'

import { marked } from 'marked'
import DOMPurify from 'isomorphic-dompurify' // Added for HTML sanitization
import { Message } from '../lib/types'

interface DeepSearchMessageProps {
  message: Message
}

export function DeepSearchMessage({ message }: DeepSearchMessageProps) {
  const isUser = message.role === 'user'
  const isError = message.role === 'bot' && message.content?.toLowerCase().startsWith('error:')

  // Configure marked with table support and better defaults
  marked.setOptions({
    breaks: true, // Enable line breaks
    gfm: true, // Explicitly enable GitHub Flavored Markdown (includes tables)
  })

  // Parse markdown and sanitize HTML
  const htmlContent = DOMPurify.sanitize(
    marked.parse(message.content || '', { async: false }) as string
  )

  const alignmentContainerClasses = isUser ? 'flex justify-end' : 'flex justify-start'

  // Unified base class and conditional classes
  const bubbleBaseClasses =
    'p-3 px-4 rounded-2xl shadow-md max-w-[85%] md:max-w-[75%] backdrop-blur-md border text-base font-sans leading-relaxed break-words'

  const userBubbleClasses = `${bubbleBaseClasses} bg-blue-500 text-white border-transparent`
  const botBubbleClasses = `${bubbleBaseClasses} bg-black dark:bg-gray-800/90 text-white dark:text-gray-100 border-none dark:border-gray-600/50` // Updated for better contrast
  const errorBubbleClasses = `${bubbleBaseClasses} bg-red-100/80 dark:bg-red-900/60 text-red-800 dark:text-red-100 border-red-300/50 dark:border-red-700/50`

  const bubbleClasses = isUser
    ? userBubbleClasses
    : isError
    ? errorBubbleClasses
    : botBubbleClasses

  // Enhanced prose classes with table styling
  const proseClasses = `prose prose-sm dark:prose-invert max-w-none 
    ${isUser ? '[&_a]:text-blue-200 hover:[&_a]:text-blue-100' : '[&_a]:text-blue-600 dark:[&_a]:text-blue-400 hover:[&_a]:text-blue-700 dark:hover:[&_a]:text-blue-300'} 
    [&_p]:my-1 first:[&_p]:mt-0 last:[&_p]:mb-0
    [&_table]:w-full [&_table]:border-collapse [&_table]:border [&_table]:border-gray-300 dark:[&_table]:border-gray-600
    [&_th]:border [&_th]:border-gray-300 dark:[&_th]:border-gray-600 [&_th]:bg-black dark:[&_th]:bg-gray-700 [&_th]:p-2 [&_th]:text-left
    [&_td]:border [&_td]:border-gray-300 dark:[_td]:border-gray-600 [&_td]:p-2
    [&_table]:overflow-x-auto` // Added table-specific styles

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