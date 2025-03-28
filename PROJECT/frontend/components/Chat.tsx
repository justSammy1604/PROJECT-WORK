'use client'

import { useState, useRef, useEffect } from "react"
import { Send, ChevronDown } from "lucide-react"
import ReactMarkdown from 'react-markdown'
import { Button } from "@/components/ui/button"
import { Input } from "./ui/input"
// Import necessary chart components from recharts
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar,
  PieChart, Pie, Cell
} from 'recharts'

// Define an extended message interface to include graph data
interface Message {
  role: 'user' | 'assistant'
  content: string
  graphData?: any[] // Assuming data structure like [{ name: 'A', value: 10 }, ...]
  graphType?: 'line' | 'bar' | 'pie' | null
}

// Define some colors for the Pie chart slices
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

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
      // Show button if not near the bottom (added a small threshold)
      const atBottom = scrollHeight - scrollTop - clientHeight <= 5
      setShowScrollButton(!atBottom)
    }

    container.addEventListener("scroll", handleScroll)
    handleScroll() // Initial check

    // Cleanup listener on component unmount
    return () => {
      if (container) {
         container.removeEventListener("scroll", handleScroll)
      }
    }
  }, []) // Rerun only if chatContainerRef changes (shouldn't often)

  // --- PARSE GRAPH DATA (No changes needed here) ---
  const parseGraphData = (response: any): Message => {
    try {
      const graphInfo = response.graphData

      if (graphInfo && graphInfo.data && graphInfo.type) {
        // Validate graph type if needed
        const validTypes = ['line', 'bar', 'pie'];
        const type = validTypes.includes(graphInfo.type) ? graphInfo.type : 'line'; // Default to line if invalid

        return {
          role: 'assistant',
          content: response.answer,
          graphData: graphInfo.data || [],
          graphType: type
        }
      }

      return {
        role: 'assistant',
        content: response.answer
      }
    } catch (error) {
      console.error('Error parsing graph data:', error)
      return {
        role: 'assistant',
        content: response.answer // Return text answer even if parsing fails
      }
    }
  }

  // --- RENDER GRAPH (Updated with Bar and Pie) ---
  const renderGraph = (message: Message) => {
    if (!message.graphData || message.graphData.length === 0 || !message.graphType) return null

    // Ensure data has the expected keys (basic check)
    const hasRequiredKeys = message.graphData.every(item => typeof item.name !== 'undefined' && typeof item.value !== 'undefined');
    if (!hasRequiredKeys) {
        console.warn("Graph data is missing 'name' or 'value' keys.");
        return <p className="text-red-500 text-sm">Graph data format is incorrect.</p>;
    }


    switch(message.graphType) {
      case 'line':
        return (
          <LineChart width={400} height={250} data={message.graphData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" activeDot={{ r: 8 }}/>
          </LineChart>
        )
      case 'bar':
        return (
          <BarChart width={400} height={250} data={message.graphData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#82ca9d" />
          </BarChart>
        )
      case 'pie':
        return (
          <PieChart width={400} height={250}>
            <Pie
              data={message.graphData}
              cx="50%" // Center pie horizontally
              cy="50%" // Center pie vertically
              labelLine={false} // Disable the line connecting label to slice (can clutter)
               // Optional: Add labels directly on or near slices
              // label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80} // Size of the pie
              fill="#8884d8" // Default fill, overridden by Cell
              dataKey="value" // The key in your data that holds the numerical value
              nameKey="name" // The key in your data that holds the name/label for the slice
            >
              {/* Map over data to assign a color to each slice */}
              {message.graphData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        )
      default:
        console.warn(`Unsupported graph type: ${message.graphType}`);
        return null // Return null for unsupported types
    }
  }

  // --- HANDLE SUBMIT (No changes needed here) ---
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
        body: JSON.stringify({
          query: input,
          extractGraphData: true
        }),
      })

      if (!response.ok) {
        // Try to read error message from response body
        let errorBody = 'Unknown error';
        try {
            errorBody = await response.text();
        } catch (_) {}
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}. Body: ${errorBody}`)
      }

      const data = await response.json()

      // Basic check for expected response structure
      if (!data || typeof data.answer === 'undefined') {
        console.error('Invalid response format received:', data);
        throw new Error('Invalid response format from server.')
      }

      const assistantMessage = parseGraphData(data)
      setMessages(prev => [...prev, assistantMessage])

    } catch (error) {
      console.error('Error fetching or processing response:', error)
      const errorContent = error instanceof Error ? error.message : 'Sorry, something went wrong.';
      const errorMessage: Message = {
        role: 'assistant',
        // Provide more context in the error message if possible
        content: `Error: ${errorContent}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // --- JSX RETURN (No changes needed in structure) ---
  return (
    <div className="flex flex-col h-[500px] max-w-2xl mx-auto border rounded shadow-md"> {/* Added some basic styling */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50"> {/* Added bg color */}
        {messages.map((message, index) => (
          <div key={index} className={`flex flex-col mb-3 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-xl shadow-sm ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-gray-800 border'}`}> {/* Adjusted styling */}
              {/* Use prose for better markdown formatting if needed: className="prose prose-sm" */}
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
            {/* Conditionally render graph container */}
            {message.graphData && message.graphType && (
              <div className="mt-2 p-2 bg-white border rounded-lg shadow-sm self-start max-w-[90%] overflow-x-auto"> {/* Graph container styling */}
                {renderGraph(message)}
              </div>
            )}
          </div>
        ))}
        {isLoading && ( // Loading indicator
            <div className="flex justify-start">
                <div className="p-3 rounded-xl bg-gray-200 text-gray-600 italic">Thinking...</div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {showScrollButton && (
        <Button
            variant="outline" // Use outline variant
            className="absolute bottom-20 right-4 rounded-full z-10 bg-white opacity-80 hover:opacity-100" // Style scroll button
            size="icon"
            onClick={scrollToBottom}
            aria-label="Scroll to bottom" // Accessibility
         >
          <ChevronDown className="h-5 w-5" />
        </Button>
      )}
      <form onSubmit={handleSubmit} className="p-4 border-t flex bg-gray-100">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about data or request a graph..." // Updated placeholder
          className="flex-1 mr-2 border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50" // Input styling
          disabled={isLoading} // Disable input while loading
        />
        <Button type="submit" disabled={isLoading || !input.trim()} aria-label="Send message"> {/* Disable if loading or empty */}
          {isLoading ? (
             // Simple loading spinner alternative
             <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>
    </div>
  )
}
