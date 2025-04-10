'use client';

import { useState } from 'react';
import React from 'react';

export default function SearchPopup() {
  const [showPopup, setShowPopup] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log('Search Query:', searchQuery);
    // Handle the search query...
  };

  const handleClose = () => {
    setShowPopup(false);
  };

  return (
    <>
      {showPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white w-80 h-80 rounded-lg shadow-lg p-6 relative flex flex-col justify-center items-center">
            <button
              onClick={handleClose}
              className="absolute top-2 right-2 text-gray-500 hover:text-red-500 text-xl"
            >
              &times;
            </button>
            <h2 className="text-lg font-semibold mb-4">Search</h2>
            <form onSubmit={handleSubmit} className="w-full flex flex-col items-center">
              <input
                type="text"
                value={searchQuery}
                onChange={handleChange}
                placeholder="Enter search..."
                className="border border-gray-300 px-4 py-2 rounded-md w-full mb-4"
              />
              <button
                type="submit"
                className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
              >
                Search
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
