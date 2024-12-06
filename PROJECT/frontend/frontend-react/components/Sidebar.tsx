import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { PlusCircle, MessageCircle, X } from 'lucide-react'

interface SidebarProps {
  isOpen: boolean;
  onClose?: () => void;  // Make onClose optional
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const chatHistory = [
    { id: 1, title: 'Previous Chat 1' },
    { id: 2, title: 'Previous Chat 2' },
    { id: 3, title: 'Previous Chat 3' },
  ]

  return (
    <div className={`fixed inset-y-0 left-0 transform ${isOpen ? 'translate-x-0' : '-translate-x-full'} transition duration-200 ease-in-out w-64 bg-gray-200 dark:bg-gray-800 p-4 flex flex-col z-10`}>
      {onClose && (  // Only render the close button if onClose is provided
        <Button 
          variant="ghost" 
          onClick={() => {
            console.log('Closing sidebar');
            onClose();
          }} 
          className="self-end mb-2 p-2 hover:bg-gray-300 dark:hover:bg-gray-700 rounded-full"
          aria-label="Close sidebar"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
      <Button className="mb-4" variant="outline">
        <PlusCircle className="mr-2 h-4 w-4" /> New Chat
      </Button>
      <ScrollArea className="flex-1">
        {chatHistory.map((chat) => (
          <Button key={chat.id} variant="ghost" className="w-full justify-start mb-2">
            <MessageCircle className="mr-2 h-4 w-4" />
            {chat.title}
          </Button>
        ))}
      </ScrollArea>
    </div>
  )
}

