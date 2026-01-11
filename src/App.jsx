import React, { useState } from 'react';
import DQNVisualization from './demos/dqn_visualization';
import RLVisualization from './demos/rl-visualization';
import RMaxVisualization from './demos/rmax-visualization';
import ValueIteration from './demos/value-iteration';

const demos = [
  {
    id: 'dqn',
    title: 'DQN: Deep Q-Network',
    description: 'Understanding Online and Target Networks',
    component: DQNVisualization,
    icon: '🤖',
    color: 'from-emerald-500 to-cyan-500'
  },
  {
    id: 'rl',
    title: 'RL Basics',
    description: 'Fundamental Reinforcement Learning Concepts',
    component: RLVisualization,
    icon: '🎯',
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'rmax',
    title: 'R-Max Algorithm',
    description: 'Exploration-Exploitation with R-Max',
    component: RMaxVisualization,
    icon: '🔍',
    color: 'from-orange-500 to-red-500'
  },
  {
    id: 'value-iteration',
    title: 'Value Iteration',
    description: 'Dynamic Programming for MDPs',
    component: ValueIteration,
    icon: '📊',
    color: 'from-blue-500 to-indigo-500'
  }
];

function App() {
  const [selectedDemo, setSelectedDemo] = useState(null);

  if (selectedDemo !== null) {
    const Demo = demos[selectedDemo].component;
    return (
      <div className="min-h-screen bg-slate-950">
        <button
          onClick={() => setSelectedDemo(null)}
          className="fixed top-4 left-4 z-50 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg border border-slate-600 transition-all flex items-center gap-2 font-semibold"
        >
          <span>←</span> Back to Menu
        </button>
        <Demo />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        * { font-family: 'Space Grotesk', sans-serif; }
      `}</style>

      <div className="container mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-emerald-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
            Interactive RL Demos
          </h1>
          <p className="text-slate-400 text-lg">
            Explore reinforcement learning concepts through interactive visualizations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {demos.map((demo, index) => (
            <button
              key={demo.id}
              onClick={() => setSelectedDemo(index)}
              className="group relative p-8 rounded-2xl border-2 border-slate-700 bg-slate-900/50 hover:bg-slate-800/50 transition-all duration-300 hover:scale-105 hover:border-slate-600 text-left"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${demo.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-300`}></div>

              <div className="relative">
                <div className="flex items-start gap-4 mb-4">
                  <div className={`text-5xl`}>
                    {demo.icon}
                  </div>
                  <div className="flex-1">
                    <h2 className={`text-2xl font-bold mb-2 bg-gradient-to-r ${demo.color} bg-clip-text text-transparent`}>
                      {demo.title}
                    </h2>
                    <p className="text-slate-400">
                      {demo.description}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-sm text-slate-500 group-hover:text-slate-400 transition-colors">
                  <span>Launch Demo</span>
                  <span className="group-hover:translate-x-1 transition-transform">→</span>
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-block p-6 rounded-xl border border-slate-700 bg-slate-900/30">
            <h3 className="text-lg font-semibold mb-2 text-slate-300">Getting Started</h3>
            <p className="text-slate-400 text-sm">
              Click on any demo card above to start exploring
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
