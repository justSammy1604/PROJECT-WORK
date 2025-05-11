'use client';

import { useState, useRef, useEffect, memo } from 'react';
import { Send, ChevronDown, Moon, Sun, Search, List, MoreHorizontal, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar,
  PieChart, Pie, Cell,
  ResponsiveContainer,
  AreaChart, Area,
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Input } from './ui/input';
import Link from 'next/link';

// Finance Topics
const financeTopics = [
  {name: 'Stock Market'},
  {name: 'Savings Plans'},
  {name: 'Invoices'},
  {name: 'Cryptocurrency'},
  {name: 'Investments'},
  {name: 'Credit Score'},
  {name: 'Market Trends'},
  {name: 'Banking'},
  {name: 'Currency Exchange'},
  {name: 'Budgeting'},
  {name: 'Cash Flow'},
  {name: 'Projections'},
];

// Constants
const PIE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
const LINE_COLOR = '#3b82f6';
const BAR_COLOR = '#10b981';
const GRID_COLOR = '#e5e7eb';
const AXIS_COLOR = '#6b7280';
const RADIAN = Math.PI / 180;

// Interfaces
interface Message {
  role: 'user' | 'assistant';
  content: string;
  graphData?: any[];
  graphType?: 'line' | 'bar' | 'pie' | null;
  className?: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

// Custom Components
const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="p-2 bg-white border border-gray-300 rounded shadow-lg text-sm">
      <p className="font-semibold text-gray-700">{label}</p>
      {payload.map((entry, index) => (
        <p key={`item-${index}`} style={{ color: entry.color || entry.payload?.fill }}>
          {`${entry.name} : ${entry.value.toLocaleString()}`}
        </p>
      ))}
    </div>
  );
};

const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  if (percent * 100 < 5) return null;

  return (
    <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={12}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

// Modal Component for Topic Selection
const TopicModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (topic: string) => void;
}> = ({ isOpen, onClose, onSubmit }) => {
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = () => {
    if (selectedTopic) {
      onSubmit(selectedTopic);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-xl max-w-md w-full border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">Select a Finance Topic</h2>
        <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
          {financeTopics.map((topic, index) => (
            <button
              key={index}
              onClick={() => setSelectedTopic(topic.name)}
              className={`p-2 rounded-md text-sm font-medium transition ${
                selectedTopic === topic.name
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {topic.name}
            </button>
          ))}
        </div>
        <div className="mt-6 flex justify-end gap-2">
          <Button variant="outline" onClick={onClose} className="rounded-md">
            Cancel
          </Button>
          <Button
            disabled={!selectedTopic}
            className="rounded-md"
            onClick={handleSubmit}
          >
            Submit
          </Button>
        </div>
      </div>
    </div>
  );
};

// Graph Rendering Component
const GraphRenderer = memo<{ message: Message }>(({ message }) => {
  if (!message.graphData || !message.graphData.length || !message.graphType) return null;

  const hasRequiredKeys = message.graphData.every(item => typeof item.name !== 'undefined' && typeof item.value !== 'undefined');
  if (!hasRequiredKeys) {
    console.warn("Graph data is missing 'name' or 'value' keys.");
    return <p className="text-red-500 text-sm">Graph data format is incorrect.</p>;
  }

  const commonProps = { margin: { top: 10, right: 30, left: 10, bottom: 10 } };
  const commonAxisProps = { stroke: AXIS_COLOR, fontSize: 12, tickLine: false };
  const commonGridProps = { stroke: GRID_COLOR, strokeDasharray: '4 4', vertical: false };
  const chartHeight = 300;

  switch (message.graphType) {
    case 'line':
      return (
        <ResponsiveContainer width="100%" height={chartHeight}>
          <AreaChart data={message.graphData} {...commonProps}>
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={LINE_COLOR} stopOpacity={0.4} />
                <stop offset="95%" stopColor={LINE_COLOR} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid {...commonGridProps} />
            <XAxis dataKey="name" {...commonAxisProps} />
            <YAxis {...commonAxisProps} axisLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: LINE_COLOR, strokeWidth: 1, strokeDasharray: '3 3' }} />
            <Legend iconSize={10} wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            <Area
              type="monotone"
              dataKey="value"
              stroke={LINE_COLOR}
              fillOpacity={1}
              fill="url(#lineGradient)"
              strokeWidth={2}
              activeDot={{ r: 6, strokeWidth: 2, fill: '#fff', stroke: LINE_COLOR }}
              dot={{ r: 3, fill: LINE_COLOR, strokeWidth: 0 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      );
    case 'bar':
      return (
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart data={message.graphData} {...commonProps} barGap={5} barCategoryGap="20%">
            <CartesianGrid {...commonGridProps} />
            <XAxis dataKey="name" {...commonAxisProps} interval={0} />
            <YAxis {...commonAxisProps} axisLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(200,200,200,0.2)' }} />
            <Legend iconSize={10} wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            <Bar dataKey="value" fill={BAR_COLOR} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
    case 'pie':
      return (
        <ResponsiveContainer width="100%" height={chartHeight}>
          <PieChart margin={{ top: 0, right: 0, left: 0, bottom: 20 }}>
            <Pie
              data={message.graphData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderCustomizedLabel}
              outerRadius={Math.min(window.innerWidth, 400) / 2 * 0.65}
              innerRadius={Math.min(window.innerWidth, 400) / 2 * 0.45}
              fill="#8884d8"
              dataKey="value"
              nameKey="name"
              paddingAngle={2}
            >
              {message.graphData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} strokeWidth={0} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle"
              iconSize={10}
              wrapperStyle={{ fontSize: '12px', bottom: 0 }}
              layout="horizontal"
              verticalAlign="bottom"
              align="center"
            />
          </PieChart>
        </ResponsiveContainer>
      );
    default:
      console.warn(`Unsupported graph type: ${message.graphType}`);
      return null;
  }
});
GraphRenderer.displayName = "GraphRenderer";

// START OF /report BUTTON FUNCTIONALITY
// Changed props to accept userQuery instead of responseText
interface AssistantMessageOptionsProps {
  userQuery: string | null; // User's query that led to this response
}

const AssistantMessageOptions: React.FC<AssistantMessageOptionsProps> = ({ userQuery }) => {
  const [showReportButton, setShowReportButton] = useState(false);

  const handleReport = async () => {
    if (!userQuery) { // Check if userQuery is null or empty
      console.error("No user query available to report or query is empty.");
      alert("Cannot report: The original user query is not available or is empty.");
      setShowReportButton(false);
      return;
    }

    const reportApiUrl = process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/report` : 'http://localhost:4200/report';

    try {
      const response = await fetch(reportApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Send user_query instead of response_text
        body: JSON.stringify({ user_query: userQuery }),
      });

      setShowReportButton(false); // Hide menu after action

      if (response.ok) {
        const data = await response.json();
        console.log('Report successful:', data.message);
        alert('Thank you! This response has been reported.'); // Updated alert
      } else {
        const errorData = await response.json().catch(() => ({error: "Failed to parse error from server"}));
        console.error('Failed to report query:', errorData.error); // Updated log
        alert(`Error reporting interaction: ${errorData.error || 'Unknown error'}`); // Updated alert
      }
    } catch (error) {
      setShowReportButton(false); // Hide menu on error
      console.error('Network error or other issue reporting query:', error); // Updated log
      alert('An error occurred while trying to report the interaction. Please check your connection.'); // Updated alert
    }
  };

  return (
    <div className="relative ml-2 self-start">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setShowReportButton(!showReportButton)}
        className="h-6 w-6 p-0 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
        aria-label="More options"
      >
        <MoreHorizontal className="h-4 w-4" />
      </Button>
      {showReportButton && (
        <div className="absolute right-0 mt-1 w-40 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-lg z-20 py-1">
          <Button
            variant="ghost"
            onClick={handleReport}
            className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 flex items-center"
            // Disable button if userQuery is not available
            disabled={!userQuery}
          >
            <AlertCircle className="h-4 w-4 mr-2 text-red-500" />
            Report inaccurate
          </Button>
        </div>
      )}
    </div>
  );
};
AssistantMessageOptions.displayName = "AssistantMessageOptions";
// END OF /report BUTTON FUNCTIONALITY

// Main Chat Component
export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isTopicModalOpen, setIsTopicModalOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [searchClicked, setSearchClicked] = useState(false);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 5);
    };

    container.addEventListener('scroll', handleScroll);
    handleScroll();
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (inputRef.current && !isLoading) {
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  }, [input, isLoading, messages]);

  const parseGraphData = (response: any): Message => {
    try {
      const graphInfo = response.graphData;
      if (graphInfo?.data && graphInfo.type) {
        const validTypes = ['line', 'bar', 'pie'];
        const type = validTypes.includes(graphInfo.type) ? graphInfo.type : 'line';
        return { role: 'assistant', content: response.answer, graphData: graphInfo.data, graphType: type };
      }
      return { role: 'assistant', content: response.answer };
    } catch (error) {
      console.error('Error parsing graph data:', error);
      return { role: 'assistant', content: response.answer };
    }
  };

  const handleTopicSubmit = async (topic: string) => {
    const linksApiUrl = process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/links` : 'http://localhost:4200/links';
    try {
      const response = await fetch(`${linksApiUrl}?search=${encodeURIComponent(topic)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => 'Unknown error');
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}. Body: ${errorBody}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.message }]);
    } catch (error) {
      console.error('Error submitting topic:', error);
      const errorContent = error instanceof Error ? error.message : 'Sorry, something went wrong.';
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${errorContent}`, className: 'text-red-500' }]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    let finalInput = input;
    if (searchClicked) {
      finalInput += ' |||TRUE||| ';
    }

    const userMessage: Message = { role: 'user', content: input }; // Store the original input before adding search marker
    setMessages(prev => [...prev, { role: 'user', content: finalInput }]); // Send finalInput with marker
    setInput('');
    setIsLoading(true);

    const queryApiUrl = process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/query` : 'http://localhost:4200/query';
    try {
      const response = await fetch(queryApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: finalInput, extractGraphData: true }),
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => 'Unknown error');
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}. Body: ${errorBody}`);
      }

      const data = await response.json();
      if (!data || typeof data.answer === 'undefined') {
        console.error('Invalid response format received:', data);
        throw new Error('Invalid response format from server.');
      }

      const assistantMessage = parseGraphData(data);
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error fetching or processing response:', error);
      const errorContent = error instanceof Error ? error.message : 'Sorry, something went wrong.';
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${errorContent}`, className: 'text-red-500' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchClick = () => {
    setSearchClicked(!searchClicked);
  };

  return (
    <div className="flex flex-col h-[500px] max-w-6xl mx-auto border rounded-lg shadow-2xl bg-white dark:bg-gray-800 dark:border-gray-700">
      <div className="absolute top-4 left-4 right-4 flex justify-between z-10">
        <Button
          variant="outline"
          size="icon"
          onClick={() => setIsTopicModalOpen(true)}
          aria-label="Select finance topic"
        >
          <List className="h-4 w-4" />
        </Button>
        <Link href="/deepsearch" passHref legacyBehavior>
            <Button
              variant="link"
              size="sm"
              className="rounded-md"
              aria-label="Go to Deep Search"
            >
              <Search className="h-4 w-4 mr-1 sm:mr-2" />
              <span className="hidden sm:inline">Deep Search</span>
            </Button>
          </Link>
        <Button
          variant="outline"
          size="icon"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
        </Button>
      </div>

      <TopicModal
        isOpen={isTopicModalOpen}
        onClose={() => setIsTopicModalOpen(false)}
        onSubmit={handleTopicSubmit}
      />

      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50 dark:bg-gray-700/50">
        {messages.map((message, index) => {
          // Determine userQuery if current message is assistant and previous is user
          let userQueryForAssistant: string | null = null;
          if (message.role === 'assistant' && index > 0 && messages[index - 1]?.role === 'user') {
            // We need the original user query, before any ' |||TRUE||| ' was appended
            // The `messages` state for user already stores the raw input if handleSubmit is adjusted
            // Let's assume `messages[index-1].content` is the query sent to the backend.
            // If `finalInput` was added to messages, we need to strip the marker.
            // For simplicity, let's assume messages[index-1].content is what we want to report.
            // If `userMessage: Message = { role: 'user', content: input };` was added to messages,
            // then `messages[index-1].content` is the original input. This seems to be the case.

            // The code `setMessages(prev => [...prev, { role: 'user', content: finalInput }]);`
            // adds `finalInput` to messages. So we need to strip it if present.
            const prevUserMessageContent = messages[index-1].content;
            const searchMarker = ' |||TRUE||| ';
            if (prevUserMessageContent.endsWith(searchMarker)) {
                userQueryForAssistant = prevUserMessageContent.substring(0, prevUserMessageContent.length - searchMarker.length);
            } else {
                userQueryForAssistant = prevUserMessageContent;
            }
          }

          return (
            <div key={index} className={`flex flex-col mb-3 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
              {message.role === 'assistant' ? (
                <div className="flex items-start max-w-[85%]">
                  <div
                    className={`p-3 rounded-2xl shadow-sm bg-white dark:bg-gray-600 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-500 ${message.className || ''}`}
                  >
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  </div>
                  {/* Pass the identified user query (original, without marker) */}
                  <AssistantMessageOptions userQuery={userQueryForAssistant} />
                </div>
              ) : (
                <div
                  className={`max-w-[85%] p-3 rounded-2xl shadow-sm bg-blue-500 text-white ${message.className || ''}`}
                >
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    {/* Display user message (could be with marker, but that's internal) */}
                    {/* For display, you might want to strip the marker here too, or ensure it's stripped before adding to messages if it shouldn't be seen */}
                    <ReactMarkdown>{message.content.replace(' |||TRUE||| ', '')}</ReactMarkdown>
                  </div>
                </div>
              )}

              {message.graphData && message.graphType && (
                <div className="mt-3 p-3 bg-white dark:bg-gray-600 border border-gray-200 dark:border-gray-500 rounded-lg shadow-sm w-full max-w-[95%] self-start overflow-hidden">
                  <GraphRenderer message={message} />
                </div>
              )}
            </div>
          );
        })}
        {isLoading && (
          <div className="flex justify-start">
            <div className="p-3 rounded-xl bg-gray-200 dark:bg-gray-500 text-gray-600 dark:text-gray-300 italic text-sm">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {showScrollButton && (
        <Button
          variant="outline"
          className="absolute bottom-20 right-4 rounded-full z-10 bg-white/80 dark:bg-gray-800/80 hover:bg-white dark:hover:bg-gray-700 backdrop-blur-sm border-gray-300 dark:border-gray-600 shadow"
          size="icon"
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
        >
          <ChevronDown className="h-5 w-5 text-gray-600 dark:text-gray-300" />
        </Button>
      )}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 dark:border-gray-700 flex bg-gray-100/80 dark:bg-gray-800/80">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about data ..."
          className="flex-1 mr-2 border-gray-300 dark:border-gray-600 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 rounded-md bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
          disabled={isLoading}
          ref={inputRef}
          key="chat-input"
        />
       <Button
        type="button"
        onClick={handleSearchClick}
        disabled={isLoading}
        aria-label="Search"
        variant={searchClicked ? 'default' : 'outline'}
        className={`mr-2 rounded-md ${searchClicked ? 'bg-black text-white' : ''}`}
        title={searchClicked ? 'Search mode on' : 'Search mode off'}
      >
        <Search className={`h-4 w-4 ${searchClicked ? 'text-white' : ''}`} />
      </Button>

        <Button
          type="submit"
          disabled={isLoading || !input.trim()}
          aria-label="Send message"
          className="rounded-md"
          onMouseDown={(e) => e.preventDefault()}
        >
          {isLoading ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>
    </div>
  );
}