// 'use client' // Keep this if you're using Next.js App Router

import { useState, useRef, useEffect } from "react"
import { Send, ChevronDown } from "lucide-react"
import ReactMarkdown from 'react-markdown'
import { Button } from "@/components/ui/button"
import { Input } from "./ui/input"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar,
  PieChart, Pie, Cell,
  ResponsiveContainer, // Import ResponsiveContainer
  AreaChart, Area // Import AreaChart/Area for gradient fill
} from 'recharts'

// Define an extended message interface
interface Message {
  role: 'user' | 'assistant'
  content: string
  graphData?: any[]
  graphType?: 'line' | 'bar' | 'pie' | null
}

// --- More Visually Appealing Colors ---
const PIE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
const LINE_COLOR = "#3b82f6"; // Blue-500 from Tailwind
const BAR_COLOR = "#10b981"; // Emerald-500 from Tailwind
const GRID_COLOR = "#e5e7eb"; // Gray-200
const AXIS_COLOR = "#6b7280"; // Gray-500

// --- Custom Tooltip Component ---
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="p-2 bg-white border border-gray-300 rounded shadow-lg text-sm">
        <p className="font-semibold text-gray-700">{`${label}`}</p>
        {payload.map((entry: any, index: number) => (
          <p key={`item-${index}`} style={{ color: entry.color || entry.payload?.fill }}>
            {`${entry.name} : ${entry.value.toLocaleString()}`} {/* Format number */}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// --- Custom Pie Label (for percentages) ---
const RADIAN = Math.PI / 180;
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, name }: any) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5; // Position label inside slice
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  // Only render label if percentage is large enough to avoid clutter
  if ((percent * 100) < 5) return null;

  return (
    <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={12}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};


export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [useAllData, setUseAllData] = useState(false); // <-- ADDED: State for the new button
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // --- Hooks and Parse Function (Keep as is) ---
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
    handleScroll() // Initial check
    return () => { if (container) { container.removeEventListener("scroll", handleScroll) } }
  }, []) // Removed container from dependency array as ref shouldn't change often

  const parseGraphData = (response: any): Message => {
    try {
      const graphInfo = response.graphData
      if (graphInfo && graphInfo.data && graphInfo.type) {
        const validTypes = ['line', 'bar', 'pie'];
        const type = validTypes.includes(graphInfo.type) ? graphInfo.type : 'line';
        // Basic validation for graph data structure (ensure it's an array)
        const data = Array.isArray(graphInfo.data) ? graphInfo.data : [];
        return { role: 'assistant', content: response.answer, graphData: data, graphType: type }
      }
      return { role: 'assistant', content: response.answer }
    } catch (error) {
      console.error('Error parsing graph data:', error)
      // Return just the answer if parsing fails
      return { role: 'assistant', content: response.answer || 'Error processing graph data.' }
    }
  }

  // --- RENDER GRAPH (Enhanced Version) ---
  const renderGraph = (message: Message) => {
    if (!message.graphData || message.graphData.length === 0 || !message.graphType) return null;

    // More robust check for required keys
    const hasRequiredKeys = message.graphData.every(item => item && typeof item.name !== 'undefined' && typeof item.value !== 'undefined');
    if (!hasRequiredKeys) {
      console.warn("Graph data is missing 'name' or 'value' keys, or item is null/undefined.");
      return <p className="text-red-500 text-sm">Graph data format is incorrect or incomplete.</p>;
    }

    // Define common axis/grid/tooltip/legend props
    const commonProps = {
        margin: { top: 10, right: 30, left: 10, bottom: 10 }, // Adjusted margins
    };
    const commonAxisProps = {
        stroke: AXIS_COLOR,
        fontSize: 12,
        tickLine: false, // Hide tick lines for cleaner look
    };
    const commonGridProps = {
        stroke: GRID_COLOR,
        strokeDasharray: "4 4", // More subtle dash
        vertical: false // Often cleaner without vertical lines
    };

    // Use ResponsiveContainer for all charts
    const chartHeight = 300; // Define a consistent height

    switch(message.graphType) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={chartHeight}>
            {/* Use AreaChart to easily add gradient fill */}
            <AreaChart data={message.graphData} {...commonProps}>
              <defs>
                <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={LINE_COLOR} stopOpacity={0.4}/>
                  <stop offset="95%" stopColor={LINE_COLOR} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid {...commonGridProps} />
              <XAxis dataKey="name" {...commonAxisProps} />
              <YAxis {...commonAxisProps} axisLine={false} /> {/* Hide Y axis line */}
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: LINE_COLOR, strokeWidth: 1, strokeDasharray: '3 3' }} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
              <Area type="monotone" dataKey="value" stroke={LINE_COLOR} fillOpacity={1} fill="url(#lineGradient)" strokeWidth={2} activeDot={{ r: 6, strokeWidth: 2, fill: '#fff', stroke: LINE_COLOR }} dot={{ r: 3, fill: LINE_COLOR, strokeWidth: 0 }}/>
            </AreaChart>
          </ResponsiveContainer>
        );
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={chartHeight}>
            <BarChart data={message.graphData} {...commonProps} barGap={5} barCategoryGap="20%">
              <CartesianGrid {...commonGridProps} />
              <XAxis dataKey="name" {...commonAxisProps} interval={0} /> {/* Show all labels if possible */}
              <YAxis {...commonAxisProps} axisLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(200,200,200,0.2)' }} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Bar dataKey="value" fill={BAR_COLOR} radius={[4, 4, 0, 0]} /> {/* Rounded top corners */}
            </BarChart>
          </ResponsiveContainer>
        );
      case 'pie':
        // Determine radius based on available width, ensuring it's not too large
        const containerWidth = chatContainerRef.current?.offsetWidth ?? 400; // Estimate width
        const maxRadius = Math.min(containerWidth * 0.8, chartHeight) / 2; // Limit radius by height too
        const outerRad = maxRadius * 0.7;
        const innerRad = maxRadius * 0.5;

        return (
          <ResponsiveContainer width="100%" height={chartHeight}>
            <PieChart margin={{ top: 5, right: 5, left: 5, bottom: 30 }}> {/* Adjust margin for legend */}
              <Pie
                data={message.graphData}
                cx="50%"
                cy="50%" // Center vertically within the allocated space
                labelLine={false}
                label={renderCustomizedLabel} // Use custom label
                outerRadius={outerRad}
                innerRadius={innerRad} // Make it a Donut chart
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                paddingAngle={message.graphData.length > 1 ? 2 : 0} // Add space between slices if more than one
              >
                {message.graphData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} strokeWidth={0} /* No border */ />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" iconSize={10} wrapperStyle={{ fontSize: '12px', bottom: 0 }} layout="horizontal" verticalAlign="bottom" align="center" />
            </PieChart>
          </ResponsiveContainer>
        );
      default:
        console.warn(`Unsupported graph type: ${message.graphType}`);
        return null;
    }
  }

  // --- HANDLE SUBMIT (MODIFIED) ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedInput = input.trim(); // Trim input once
    if (!trimmedInput) return

    const userMessage: Message = { role: 'user', content: trimmedInput }
    setMessages(prev => [...prev, userMessage])

    // Capture the current setting for 'useAllData' for *this* specific request
    const currentUseAllDataSetting = useAllData;

    // Reset input and the 'useAllData' flag for the *next* interaction
    setInput('')
    setUseAllData(false); // <-- RESET useAllData STATE HERE after capturing
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:4200/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify({
            query: trimmedInput, // Use the trimmed input
            extractGraphData: true,
            useAllData: currentUseAllDataSetting // <-- ADDED: Send the captured state
        }),
      })

      if (!response.ok) {
        let errorBody = 'Unknown error';
        try {
             // Try to get more specific error from backend response body
             const errorData = await response.json();
             errorBody = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch (_) {
            // If response is not JSON or empty, fallback to text
            try { errorBody = await response.text(); } catch { /* ignore further errors */ }
        }
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}. Details: ${errorBody}`)
      }

      const data = await response.json()
      // Add more robust checking for the response structure
      if (!data || typeof data.answer === 'undefined') {
        console.error('Invalid response format received:', data);
        throw new Error('Invalid response format from server (missing "answer").')
      }

      const assistantMessage = parseGraphData(data)
      setMessages(prev => [...prev, assistantMessage])

    } catch (error) {
      console.error('Error fetching or processing response:', error)
      // Display a more user-friendly error message
      const errorContent = error instanceof Error ? error.message : 'Sorry, an unexpected error occurred.';
      const errorMessage: Message = { role: 'assistant', content: `Error: ${errorContent}` }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    // Adjusted height to better accommodate the extra button, use h-screen or flex-grow in a parent if needed
    <div className="flex flex-col h-[550px] max-w-6xl mx-auto border rounded-lg shadow-lg bg-white">
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50">
        {messages.map((message, index) => (
          <div key={index} className={`flex flex-col mb-3 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
            {/* Message Bubble */}
            <div className={`max-w-[85%] p-3 rounded-2xl shadow-sm ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-gray-800 border border-gray-200'}`}>
              <div className="prose prose-sm max-w-none dark:prose-invert break-words"> {/* Added break-words */}
                <ReactMarkdown>
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
            {/* Graph Area */}
            {message.graphData && message.graphType && message.graphData.length > 0 && ( // Added check for graphData length
               <div className="mt-3 p-3 bg-white border border-gray-200 rounded-lg shadow-sm w-full max-w-[95%] self-start overflow-hidden"> {/* Ensure width is available */}
                {renderGraph(message)}
              </div>
            )}
          </div>
        ))}
        {/* Loading Indicator */}
        {isLoading && (
            <div className="flex justify-start">
                <div className="p-3 rounded-xl bg-gray-200 text-gray-600 italic text-sm">Thinking...</div>
            </div>
        )}
        {/* Scroll Anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll To Bottom Button */}
      {showScrollButton && (
        <Button variant="outline" className="absolute bottom-[80px] right-4 rounded-full z-10 bg-white/80 hover:bg-white backdrop-blur-sm border-gray-300 shadow" size="icon" onClick={scrollToBottom} aria-label="Scroll to bottom"> {/* Adjusted bottom position */}
          <ChevronDown className="h-5 w-5 text-gray-600" />
        </Button>
      )}

      {/* FORM AREA - MODIFIED */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 flex flex-col bg-gray-100/80 gap-2"> {/* Added gap */}
        {/* Input and Send Button Row */}
        <div className="flex w-full gap-2"> {/* Added gap */}
            <Input
              type="text" value={input} onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about data or request a graph..."
              className="flex-1 border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 rounded-md"
              disabled={isLoading}
              aria-label="Chat input"
            />
            <Button type="submit" disabled={isLoading || !input.trim()} aria-label="Send message" className="rounded-md shrink-0"> {/* Added shrink-0 */}
              {isLoading ? ( <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div> ) : ( <Send className="h-4 w-4" /> )}
            </Button>
        </div>

        {/* 'Use All Data' Button - ADDED */}
        <Button
            type="button"
            variant={useAllData ? "default" : "outline"} // Change variant when active
            size="sm"
            className="w-full sm:w-auto sm:self-start" 
            onClick={() => setUseAllData(true)}
            disabled={isLoading}
            aria-pressed={useAllData} // For accessibility
        >
            {useAllData ? "Using All Available Data" : "Use All Data"}
        </Button>
      </form>
    </div>
  )
}