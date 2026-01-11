import React, { useState } from 'react';
import DQNVisualization from './demos/dqn_visualization';
import RLVisualization from './demos/rl-visualization';
import RMaxVisualization from './demos/rmax-visualization';
import ValueIteration from './demos/value-iteration';
import PolicyIteration from './demos/policy-iteration';
import BayesianOptimization from './demos/bayesian-optimization';
import BayesianRegression from './demos/bayesian_regression';
import GPVisualization from './demos/gp-visualization';
import VariationalInference from './demos/variational-inference';

const demos = [
  {
    id: 'bayesian-regression',
    title: 'Bayesian Linear Regression',
    description: 'Probabilistic Approach to Linear Models',
    component: BayesianRegression,
    icon: '📈',
    color: 'from-teal-500 to-cyan-500'
  },
  {
    id: 'gp-visualization',
    title: 'Gaussian Processes',
    description: 'Non-parametric Bayesian Models',
    component: GPVisualization,
    icon: '📉',
    color: 'from-rose-500 to-pink-500'
  },
  {
    id: 'variational-inference',
    title: 'Variational Inference',
    description: 'Approximate Bayesian Inference via Optimization',
    component: VariationalInference,
    icon: '🔬',
    color: 'from-violet-500 to-purple-500'
  },
  {
    id: 'bayesian-optimization',
    title: 'Bayesian Optimization',
    description: 'Efficient Global Optimization with Gaussian Processes',
    component: BayesianOptimization,
    icon: '🎲',
    color: 'from-yellow-500 to-orange-500'
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
    id: 'policy-iteration',
    title: 'Policy Iteration',
    description: 'Iterative Policy Evaluation and Improvement',
    component: PolicyIteration,
    icon: '🔄',
    color: 'from-sky-500 to-blue-500'
  },
  {
    id: 'value-iteration',
    title: 'Value Iteration',
    description: 'Dynamic Programming for MDPs',
    component: ValueIteration,
    icon: '📊',
    color: 'from-blue-500 to-indigo-500'
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
    id: 'dqn',
    title: 'DQN: Deep Q-Network',
    description: 'Understanding Online and Target Networks',
    component: DQNVisualization,
    icon: '🤖',
    color: 'from-emerald-500 to-cyan-500'
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
            PAI Demos
          </h1>
          <p className="text-slate-400 text-lg mb-3">
            Explore machine learning and reinforcement learning concepts through interactive visualizations
          </p>
          <p className="text-slate-500 text-sm">
            Author: <span className="text-slate-400 font-semibold">mmeciani</span>
          </p>
        </div>

        <div className="max-w-4xl mx-auto mb-8 p-4 rounded-lg border border-amber-500/30 bg-amber-500/5">
          <div className="flex items-start gap-3">
            <span className="text-amber-500 text-xl mt-0.5">⚠️</span>
            <div>
              <h3 className="text-amber-400 font-semibold mb-1">Disclaimer</h3>
              <p className="text-slate-400 text-sm">
                This material was fully AI-generated. There is no guarantee of the correctness of the content presented in these demonstrations.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
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
