import React from 'react';
// import { useEffect, useRef, useState } from 'react';
// import LangflowClient from './LangflowClient'; 
import './App.css';

function App() {
  // const chatRef = useRef(null);
  // const [messages, setMessages] = useState([]);
  // const langflowClient = new LangflowClient('http://127.0.0.1:7860', 'MY9hUpoILW4B6IWnd0OeOxkzV-dgn8sv'); 
  const flowIdOrName = '697fb9c8-9f17-46e2-b61d-819c5dd2be61'; 
  // const [chatbotData, setChatbotData] = useState(null);

  // useEffect(() => {
  //   // Fetch Chatbot_cohere.json data
  //   async function fetchChatbotData() {
  //     try {
  //       const response = await fetch('/Chatbot_cohere.json'); // Path relative to public folder
  //       if (!response.ok) {
  //         throw new Error('Failed to fetch Chatbot_cohere.json');
  //       }
  //       const data = await response.json();
  //       setChatbotData(data); // Store fetched data in state
  //     } catch (error) {
  //       console.error('Error fetching Chatbot_cohere.json:', error);
  //     }
  //   }

  //   fetchChatbotData();
  // }, []);

  // // Handle sending user messages
  // const handleSendMessage = async (inputValue) => {
  //   try {
  //     const response = await langflowClient.runFlow(
  //       flowIdOrName,
  //       inputValue,
  //       chatbotData?.data?.tweaks ?? {}, // Use tweaks from Chatbot_cohere.json if available
  //       false, // Set to true if you want to enable streaming
  //       (data) => {
  //         setMessages((prevMessages) => [...prevMessages, data.chunk]);
  //       }, // onUpdate
  //       (message) => {
  //         console.log('Stream Closed:', message);
  //       }, // onClose
  //       (error) => {
  //         console.error('Stream Error:', error);
  //       } // onError
  //     );

  //     if (response?.outputs?.length > 0) {
  //       const outputMessage = response.outputs[0]?.outputs[0]?.outputs?.message?.message?.text;
  //       if (outputMessage) {
  //         setMessages((prevMessages) => [...prevMessages, outputMessage]);
  //       }
  //     }
  //   } catch (error) {
  //     console.error('Error sending message:', error);
  //   }
  // };

  return (
    <div className="App">
      <h1 className="text-3xl font-bold text-center my-8">Your Main App Content</h1>
      {/* <div className="chat-container" ref={chatRef} > */}
      <langflow-chat
          window_title="Chatbot_cohere"
          flow_id={flowIdOrName}
          host_url="http://localhost:7860"
          chat_position="top-left"
          
          height={500}
          width={400}
        ></langflow-chat>
{/* </div> */}

      {/* <div>
        {messages.map((msg, index) => (
          <div key={index}>{msg}</div>
        ))}
         <button onClick={() => handleSendMessage('User message')}>Send Message</button>
      </div> */}
    </div>
  );
}

export default App;
