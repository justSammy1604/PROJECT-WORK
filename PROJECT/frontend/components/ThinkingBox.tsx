import { useState } from 'react'

interface ThinkingBoxProps {
  content: string
}

export function ThinkingBox({ content }: ThinkingBoxProps) {
  const [isEnlarged, setIsEnlarged] = useState(false)

  return (
    <div
      className={`p-4 rounded-2xl bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg shadow-lg border border-gray-200/50 dark:border-gray-700/50 transition-all duration-300 ${
        isEnlarged
          ? 'fixed inset-4 z-50 max-w-4xl mx-auto h-auto overflow-auto'
          : 'max-w-[85%] lg:max-w-[75%]'
      } overflow-x-hidden overflow-wrap-break-word`}
      onClick={() => setIsEnlarged(!isEnlarged)}
      role="button"
      tabIndex={0}
      aria-label={isEnlarged ? 'Shrink thinking box' : 'Enlarge thinking box'}
    >
      <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Thinking Process
      </h3>
      <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words">
        {content}
      </pre>
      {isEnlarged && (
        <button
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-full text-sm font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 transition-colors"
          onClick={(e) => {
            e.stopPropagation()
            setIsEnlarged(false)
          }}
        >
          Close
        </button>
      )}
    </div>
  )
}