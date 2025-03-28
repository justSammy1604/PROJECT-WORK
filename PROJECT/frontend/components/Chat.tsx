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
    handleScroll()
    return () => { if (container) { container.removeEventListener("scroll", handleScroll) } }
  }, [])

  const parseGraphData = (response: any): Message => {
    try {
      const graphInfo = response.graphData
      if (graphInfo && graphInfo.data && graphInfo.type) {
        const validTypes = ['line', 'bar', 'pie'];
        const type = validTypes.includes(graphInfo.type) ? graphInfo.type : 'line';
        return { role: 'assistant', content: response.answer, graphData: graphInfo.data || [], graphType: type }
      }
      return { role: 'assistant', content: response.answer }
    } catch (error) {
      console.error('Error parsing graph data:', error)
      return { role: 'assistant', content: response.answer }
    }
  }

  // --- RENDER GRAPH (Enhanced Version) ---
  const renderGraph = (message: Message) => {
    if (!message.graphData || message.graphData.length === 0 || !message.graphType) return null;

    const hasRequiredKeys = message.graphData.every(item => typeof item.name !== 'undefined' && typeof item.value !== 'undefined');
    if (!hasRequiredKeys) {
      console.warn("Graph data is missing 'name' or 'value' keys.");
      return <p className="text-red-500 text-sm">Graph data format is incorrect.</p>;
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
        return (
          <ResponsiveContainer width="100%" height={chartHeight}>
            <PieChart margin={{ top: 0, right: 0, left: 0, bottom: 20 }}> {/* Adjust margin for legend */}
              <Pie
                data={message.graphData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomizedLabel} // Use custom label
                outerRadius={Math.min(window.innerWidth, 400) / 2 * 0.65 } // Responsive radius
                innerRadius={Math.min(window.innerWidth, 400) / 2 * 0.45 } // Make it a Donut chart
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                paddingAngle={2} // Add space between slices
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

  // --- HANDLE SUBMIT (Keep as is) ---
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
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify({ query: input, extractGraphData: true }),
      })

      if (!response.ok) {
        let errorBody = 'Unknown error'; try { errorBody = await response.text(); } catch (_) {}
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}. Body: ${errorBody}`)
      }
      const data = await response.json()
      if (!data || typeof data.answer === 'undefined') {
        console.error('Invalid response format received:', data);
        throw new Error('Invalid response format from server.')
      }
      const assistantMessage = parseGraphData(data)
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error fetching or processing response:', error)
      const errorContent = error instanceof Error ? error.message : 'Sorry, something went wrong.';
      const errorMessage: Message = { role: 'assistant', content: `Error: ${errorContent}` }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[500px] max-w-6xl mx-auto border rounded-lg shadow-lg bg-white">
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50"> 
        {messages.map((message, index) => (
          <div key={index} className={`flex flex-col mb-3 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-2xl shadow-sm ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white text-gray-800 border border-gray-200'}`}>
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>
                {message.content}
              </ReactMarkdown>
            </div>            
            </div>
            {message.graphData && message.graphType && (
              // Container for the graph - ensures responsiveness works
               <div className="mt-3 p-3 bg-white border border-gray-200 rounded-lg shadow-sm w-full max-w-[95%] self-start overflow-hidden"> {/* Ensure width is available */}
                {renderGraph(message)}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start">
                <div className="p-3 rounded-xl bg-gray-200 text-gray-600 italic text-sm">Thinking...</div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {showScrollButton && (
        <Button variant="outline" className="absolute bottom-20 right-4 rounded-full z-10 bg-white/80 hover:bg-white backdrop-blur-sm border-gray-300 shadow" size="icon" onClick={scrollToBottom} aria-label="Scroll to bottom">
          <ChevronDown className="h-5 w-5 text-gray-600" />
        </Button>
      )}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 flex bg-gray-100/80">
        <Input
          type="text" value={input} onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about data or request a graph..."
          className="flex-1 mr-2 border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 rounded-md" // Slightly rounded input
          disabled={isLoading}
        />
        <Button type="submit" disabled={isLoading || !input.trim()} aria-label="Send message" className="rounded-md"> {/* Matching rounded button */}
          {isLoading ? ( <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div> ) : ( <Send className="h-4 w-4" /> )}
        </Button>
      </form>
    </div>
  )
}
