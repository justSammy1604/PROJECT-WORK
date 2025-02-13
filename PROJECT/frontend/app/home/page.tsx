'use client';
import React from 'react';
import { Brain, FileText, Zap, Lock } from 'lucide-react';
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
      <span className="border-r-2 border-black ml-1 animate-cursor-blink">|</span>
    </span>
  );
};

// HeroSection Component
const HeroSection = () => {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 bg-gradient-to-b from-[#F1F1F1] to-[#F5F5DC]">
        <h1 className="text-4xl md:text-6xl font-bold text-black mb-6 text-center">
          Meet Your AI Research Assistant
        </h1>
        <div className="text-xl md:text-2xl text-black-500/80 mb-8 h-20 text-center">
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
            className="bg-black text-white hover:bg-black-700 text-lg px-8 py-6 rounded-full transform transition-all hover:scale-105 shadow-lg hover:shadow-xl"
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
      description: "Advanced AI processing for deep understanding of your data"
    },
    {
      icon: <FileText className="w-8 h-8 mb-4" />,
      title: "RAG Technology",
      description: "Retrieval-Augmented Generation for accurate, contextual responses"
    },
    {
      icon: <Zap className="w-8 h-8 mb-4" />,
      title: "Fast Processing",
      description: "Quick analysis and response generation"
    },
    {
      icon: <Lock className="w-8 h-8 mb-4" />,
      title: "Reliable",
      description: "Gives you reliable answers to your queries"
    }
  ];

  return (
    <div className="py-20 bg-black-100">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl md:text-4xl font-bold text-black text-center mb-16">
          Powerful Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="flex flex-col items-center text-center p-6 bg-white rounded-lg shadow-lg hover:shadow-xl duration-300 transform transition-all hover:-translate-y-2"
            >
              {feature.icon}
              <h3 className="text-xl font-semibold mb-2 text-black">{feature.title}</h3>
              <p className="text-black">{feature.description}</p>
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
      description: "Simply upload your documents, research papers, or data"
    },
    {
      number: "02",
      title: "AI Processing",
      description: "Our AI analyzes and indexes your content for quick retrieval"
    },
    {
      number: "03",
      title: "Ask Questions",
      description: "Ask anything related to your data and get accurate answers"
    }
  ];

  return (
    <div className="py-20 bg-black-50">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl md:text-4xl font-bold text-blue text-center mb-16">
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <div
              key={index}
              className="relative p-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300"
            >
              <span className="absolute top-6 right-6 text-4xl font-bold text-blue-100">
                {step.number}
              </span>
              <h3 className="text-xl font-semibold mb-3 mt-4 text-blue">
                {step.title}
              </h3>
              <p className="text-black">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Main Index Component
const Index = () => {
  return (
    <div className="min-h-screen bg-white">
      <HeroSection />
      <Features />
      <HowItWorks />
    </div>
  );
};

export default Index;
