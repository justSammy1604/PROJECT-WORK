export interface Message {
  role: 'user' | 'bot' | 'thinking'
  content: string
  }