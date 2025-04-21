'use client';

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

export default function grid() {
  return (
    <div className="bg-[#6fa4f5] p-6 rounded-lg shadow-lg text-black">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {financeTopics.map((topic, index) => (
          <button
            key={index}
            className="flex items-center gap-2 bg-[#ffffff] hover:bg-[#454584] px-4 py-2 rounded-md text-sm font-medium transition"
          >
            <span className="text-lg"></span>
            <span>{topic.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
