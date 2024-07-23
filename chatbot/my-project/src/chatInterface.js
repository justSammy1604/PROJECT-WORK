// import React, { useState, useRef, useEffect } from 'react';
// import { X, Send } from 'lucide-react';
// import LangflowClient from './LangflowClient'; // Ensure LangflowClient.js is in the same directory or update the import path
// import './App.css'

// const ChatInterface = () => {
//   const [isOpen, setIsOpen] = useState(false);
//   const [messages, setMessages] = useState([]);
//   const [inputValue, setInputValue] = useState('');
//   const messagesEndRef = useRef(null);
//   const langflowClient = useRef(new LangflowClient('http://192.168.23.236:7880', 'MY9hUpoILW4B6IWnd0OeOxkzV-dgn8sv'));

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(scrollToBottom, [messages]);

//   const toggleChat = () => setIsOpen(!isOpen);

//   const handleInputChange = (e) => setInputValue(e.target.value);

//   const handleSendMessage = async () => {
//     if (inputValue.trim()) {
//       setMessages([...messages, { text: inputValue, isUser: true }]);
//       const userMessage = inputValue;
//       setInputValue('');

//       try {
//         const response = await langflowClient.current.runFlow(
//           '697fb9c8-9f17-46e2-b61d-819c5dd2be61',
//           userMessage,
//           {},
//           false,
//           (data) => console.log("Received:", data.chunk), // onUpdate
//           (message) => console.log("Stream Closed:", message), // onClose
//           (error) => console.log("Stream Error:", error) // onError
//         );

//         const flowOutputs = response.outputs[0];
//         const firstComponentOutputs = flowOutputs.outputs[0];
//         const output = firstComponentOutputs.outputs.message;

//         setMessages(prev => [...prev, { text: output.message.text, isUser: false }]);
//       } catch (error) {
//         console.error('Error running flow:', error);
//         setMessages(prev => [...prev, { text: 'Error: Unable to get response', isUser: false }]);
//       }
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter') {
//       handleSendMessage();
//     }
//   };

//   return (
//     <div className="relative h-screen">
//       {!isOpen && (
//         <button
//           onClick={toggleChat}
//           className="fixed bottom-5 right-5 bg-gray-300 text-black hover:bg-black hover:text-white transition-colors duration-300 py-2 px-4 rounded"
//         >
//           Chat
//         </button>
//       )}

//       <div className={`fixed bottom-0 right-0 w-72 h-4/5 bg-gray-100 shadow-lg transition-transform duration-300 ease-in-out transform ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
//         <div className="bg-green-600 text-white p-2 flex justify-between items-center rounded-t-lg">
//           <h3 className="text-lg font-semibold">Chat</h3>
//           <button onClick={toggleChat} className="text-white hover:text-black transition-colors">
//             <X size={24} />
//           </button>
//         </div>

//         <div className="h-[calc(100%-110px)] overflow-y-auto p-4">
//           {messages.map((msg, index) => (
//             <div key={index} className={`mb-2 ${msg.isUser ? 'text-right' : 'text-left'}`}>
//               <span className={`inline-block p-2 rounded-lg ${msg.isUser ? 'bg-green-200' : 'bg-white'}`}>
//                 {msg.text}
//               </span>
//             </div>
//           ))}
//           <div ref={messagesEndRef} />
//         </div>

//         <div className="absolute bottom-0 left-0 right-0 flex">
//           <input
//             type="text"
//             value={inputValue}
//             onChange={handleInputChange}
//             onKeyPress={handleKeyPress}
//             className="flex-grow p-2 border-t border-gray-300"
//             placeholder="Type your message..."
//           />
//           <button
//             onClick={handleSendMessage}
//             className={`bg-green-600 text-white p-2 ${inputValue.trim() ? 'hover:bg-green-700' : ''} transition-colors`}
//           >
//             <Send size={24} />
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default ChatInterface;
