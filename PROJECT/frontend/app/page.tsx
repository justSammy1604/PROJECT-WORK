'use client';
import React, { useState } from 'react';
import { Brain, FileText, Zap, Lock, Sun, Moon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

// AnimatedText Component
const AnimatedText = ({ texts, className = "" }: { texts: string[], className?: string }) => {
  const [currentTextIndex, setCurrentTextIndex] = React.useState(0);
  const [displayText, setDisplayText] = React.useState("");
  const [isTyping, setIsTyping] = React.useState(true);

  React.useEffect(() => {
    let timeout: NodeJS.Timeout;
    
    if (isTyping) {
      if (displayText.length < texts[currentTextIndex].length) {
        timeout = setTimeout(() => {
          setDisplayText(texts[currentTextIndex].slice(0, displayText.length + 1));
        }, 100);
      } else {
        timeout = setTimeout(() => {
          setIsTyping(false);
        }, 2000);
      }
    } else {
      if (displayText.length > 0) {
        timeout = setTimeout(() => {
          setDisplayText(displayText.slice(0, -1));
        }, 50);
      } else {
        setCurrentTextIndex((prev) => (prev + 1) % texts.length);
        setIsTyping(true);
      }
    }

    return () => clearTimeout(timeout);
  }, [displayText, isTyping, currentTextIndex, texts]);

  return (
    <span className={`${className} inline-block`}>
      {displayText}
      <span className="border-r-2 border-black dark:border-white ml-1 animate-cursor-blink">|</span>
    </span>
  );
};

// HeroSection Component
const HeroSection = ({ toggleTheme, isDark }: { toggleTheme: () => void, isDark: boolean }) => {
  return (
    <div className="relative min-h-[80vh] flex flex-col items-center justify-center px-4 bg-gradient-to-b from-[#F1F1F1] via-[#F5F5DC] to-gray-100 dark:from-[#1a1a1a] dark:via-[#2d2d2d] dark:to-black overflow-hidden">
      {/* Toggle Button */}
      <button
        onClick={toggleTheme}
        className="absolute top-4 right-4 p-2 rounded-full bg-gray-200 dark:bg-black text-black dark:text-white hover:bg-gray-300 dark:hover:bg-black transition-colors z-10"
        aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      >
        {isDark ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
      </button>
      <h1 className="text-4xl md:text-6xl font-bold text-black dark:text-white mb-6 text-center z-10">
        Meet Your AI Research Assistant
      </h1>
      <div className="text-xl md:text-2xl text-black/80 dark:text-gray-300 mb-8 h-20 text-center z-10">
        <AnimatedText
          texts={[
            "I can help you analyze complex data...",
            "I can assist with your research...",
            "I learn from your documents...",
            "I provide accurate, contextual answers...",
          ]}
        />
      </div>
      <Link href="/home" passHref>
        <Button
          className="bg-black text-white dark:bg-white dark:text-black hover:bg-black dark:hover:bg-gray-200 text-lg px-8 py-6 rounded-full transform transition-all hover:scale-105 shadow-lg hover:shadow-xl z-10"
        >
          Start Chatting Now
        </Button>
      </Link>
    </div>
  );
};

// Features Component
const Features = () => {
  const features = [
    {
      icon: <Brain className="w-8 h-8 mb-4" />,
      title: "Intelligent Analysis",
      description: "Advanced AI processing for deep understanding of your data",
    },
    {
      icon: <FileText className="w-8 h-8 mb-4" />,
      title: "RAG Technology",
      description: "Retrieval-Augmented Generation for accurate, contextual responses",
    },
    {
      icon: <Zap className="w-8 h-8 mb-4" />,
      title: "Fast Processing",
      description: "Quick analysis and response generation",
    },
    {
      icon: <Lock className="w-8 h-8 mb-4" />,
      title: "Reliable",
      description: "Gives you reliable answers to your queries",
    },
  ];

  return (
    <div className="py-20 bg-gradient-to-b from-gray-100 via-gray-100 to-gray-50 dark:from-gray-900 dark:via-gray-900 dark:to-black">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl md:text-4xl font-bold text-black dark:text-white text-center mb-16">
          Powerful Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="flex flex-col items-center text-center p-6 bg-white dark:bg-gray-900 rounded-lg shadow-lg hover:shadow-xl duration-300 transform transition-all hover:-translate-y-2"
            >
              {feature.icon}
              <h3 className="text-xl font-semibold mb-2 text-black dark:text-white">{feature.title}</h3>
              <p className="text-black dark:text-gray-300">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// HowItWorks Component
const HowItWorks = () => {
  const steps = [
    {
      number: "01",
      title: "Upload Your Data",
      description: "Simply upload your documents, research papers, or data",
    },
    {
      number: "02",
      title: "AI Processing",
      description: "Our AI analyzes and indexes your content for quick retrieval",
    },
    {
      number: "03",
      title: "Ask Questions",
      description: "Ask anything related to your data and get accurate answers",
    },
  ];

  return (
    <div className="py-20 bg-white  dark:bg-black">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl md:text-4xl font-bold text-blue-600 dark:text-blue-300 text-center mb-16">
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <div
              key={index}
              className="relative p-6 bg-white dark:bg-gray-700 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300"
            >
              <span className="absolute top-6 right-6 text-4xl font-bold text-blue-300 dark:text-blue-200">
                {step.number}
              </span>
              <h3 className="text-xl font-semibold mb-3 mt-4 text-blue-600 dark:text-blue-300">
                {step.title}
              </h3>
              <p className="text-black dark:text-gray-300">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Footer in Index Component
const Index = () => {
  const [isDark, setIsDark] = useState(false);

  const toggleTheme = () => {
    setIsDark((prev) => !prev);
  };

  return (
    <div className={`min-h-screen ${isDark ? 'dark' : ''}`} data-theme={isDark ? 'dark' : 'light'}>
      <div className="bg-white transition-colors duration-300 dark:bg-black">
        <HeroSection toggleTheme={toggleTheme} isDark={isDark} />
        <Features />
        <HowItWorks />
        {/* Powerful Sun Rays and Luminance Effect at Bottom */}
        <footer className="relative py-20 bg-gradient-to-t from-transparent dark:from-transparent dark:to-black overflow-hidden">
{/*           <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-[150%] h-96 bg-gradient-radial from-yellow-400/60 via-yellow-300/30 to-transparent opacity-70 pointer-events-none animate-pulse">
            {/* Enhanced Sun Rays */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_#ffcb00_0%,_transparent_50%)] opacity-50"></div>
            <div className="absolute inset-0 bg-[conic-gradient(from_0deg_at_50%_50%,_#ffcb00_0deg,_transparent_30deg,_transparent_330deg,_#ffcb00_360deg)] opacity-40 rotate-45"></div>
            {/* Luminance Glow */}
            <div className="absolute bottom-0 w-full h-40 bg-gradient-to-t from-yellow-200/60 dark:from-yellow-900/60 to-transparent"></div>
          </div> */}
        </footer>
      </div>
    </div>
  );
};
export default Index;
