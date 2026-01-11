import { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 4;
const GAMMA = 0.9;
const THETA = 0.001;

const ACTIONS = [
  { name: 'up', dx: 0, dy: -1, arrow: '↑' },
  { name: 'right', dx: 1, dy: 0, arrow: '→' },
  { name: 'down', dx: 0, dy: 1, arrow: '↓' },
  { name: 'left', dx: -1, dy: 0, arrow: '←' }
];

const CELL_TYPES = {
  NORMAL: 'normal',
  GOAL: 'goal',
  PIT: 'pit',
  START: 'start'
};

const initialGrid = () => {
  const grid = Array(GRID_SIZE).fill(null).map(() => 
    Array(GRID_SIZE).fill(CELL_TYPES.NORMAL)
  );
  grid[0][0] = CELL_TYPES.START;
  grid[GRID_SIZE-1][GRID_SIZE-1] = CELL_TYPES.GOAL;
  grid[1][2] = CELL_TYPES.PIT;
  grid[2][1] = CELL_TYPES.PIT;
  return grid;
};

const getReward = (cellType) => {
  switch(cellType) {
    case CELL_TYPES.GOAL: return 10;
    case CELL_TYPES.PIT: return -10;
    default: return -0.1;
  }
};

const isTerminal = (cellType) => cellType === CELL_TYPES.GOAL || cellType === CELL_TYPES.PIT;

export default function PolicyIterationViz() {
  const [grid] = useState(initialGrid);
  const [values, setValues] = useState(() => 
    Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0))
  );
  const [policy, setPolicy] = useState(() => 
    Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0))
  );
  const [phase, setPhase] = useState('idle');
  const [iteration, setIteration] = useState(0);
  const [evalSweeps, setEvalSweeps] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [highlightedCell, setHighlightedCell] = useState(null);
  const [policyStable, setPolicyStable] = useState(false);
  const [history, setHistory] = useState([]);

  const getNextState = useCallback((x, y, action) => {
    const nx = Math.max(0, Math.min(GRID_SIZE - 1, x + action.dx));
    const ny = Math.max(0, Math.min(GRID_SIZE - 1, y + action.dy));
    return { nx, ny };
  }, []);

  const policyEvaluationStep = useCallback(() => {
    let maxDelta = 0;
    const newValues = values.map(row => [...row]);
    
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (isTerminal(grid[y][x])) {
          newValues[y][x] = getReward(grid[y][x]);
          continue;
        }
        
        const action = ACTIONS[policy[y][x]];
        const { nx, ny } = getNextState(x, y, action);
        const reward = getReward(grid[ny][nx]);
        const newValue = reward + GAMMA * values[ny][nx];
        maxDelta = Math.max(maxDelta, Math.abs(newValue - values[y][x]));
        newValues[y][x] = newValue;
      }
    }
    
    setValues(newValues);
    return maxDelta < THETA;
  }, [values, policy, grid, getNextState]);

  const policyImprovement = useCallback(() => {
    let stable = true;
    const newPolicy = policy.map(row => [...row]);
    
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (isTerminal(grid[y][x])) continue;
        
        const oldAction = policy[y][x];
        let bestValue = -Infinity;
        let bestAction = 0;
        
        for (let a = 0; a < ACTIONS.length; a++) {
          const action = ACTIONS[a];
          const { nx, ny } = getNextState(x, y, action);
          const reward = getReward(grid[ny][nx]);
          const value = reward + GAMMA * values[ny][nx];
          
          if (value > bestValue) {
            bestValue = value;
            bestAction = a;
          }
        }
        
        newPolicy[y][x] = bestAction;
        if (oldAction !== bestAction) stable = false;
      }
    }
    
    setPolicy(newPolicy);
    return stable;
  }, [values, policy, grid, getNextState]);

  const step = useCallback(() => {
    if (policyStable) return;
    
    if (phase === 'idle' || phase === 'improvement') {
      setPhase('evaluation');
      setEvalSweeps(0);
      setHistory(h => [...h, { type: 'start_eval', iteration: iteration + 1 }]);
    } else if (phase === 'evaluation') {
      const converged = policyEvaluationStep();
      setEvalSweeps(s => s + 1);
      
      if (converged) {
        setPhase('improvement');
        setHistory(h => [...h, { type: 'eval_done', sweeps: evalSweeps + 1 }]);
      }
    }
    
    if (phase === 'improvement') {
      const stable = policyImprovement();
      setIteration(i => i + 1);
      setPolicyStable(stable);
      
      if (stable) {
        setPhase('done');
        setHistory(h => [...h, { type: 'done' }]);
        setIsRunning(false);
      } else {
        setHistory(h => [...h, { type: 'improved' }]);
      }
    }
  }, [phase, policyStable, policyEvaluationStep, policyImprovement, iteration, evalSweeps]);

  useEffect(() => {
    if (!isRunning || policyStable) return;
    const timer = setTimeout(step, speed);
    return () => clearTimeout(timer);
  }, [isRunning, step, speed, policyStable]);

  const reset = () => {
    setValues(Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0)));
    setPolicy(Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0)));
    setPhase('idle');
    setIteration(0);
    setEvalSweeps(0);
    setIsRunning(false);
    setPolicyStable(false);
    setHistory([]);
    setHighlightedCell(null);
  };

  const getValueColor = (value) => {
    const normalized = Math.max(-1, Math.min(1, value / 10));
    if (normalized > 0) {
      return `rgba(34, 197, 94, ${Math.abs(normalized) * 0.7 + 0.1})`;
    } else if (normalized < 0) {
      return `rgba(239, 68, 68, ${Math.abs(normalized) * 0.7 + 0.1})`;
    }
    return 'rgba(100, 116, 139, 0.2)';
  };

  const getCellStyle = (cellType) => {
    switch(cellType) {
      case CELL_TYPES.GOAL: return 'bg-emerald-500/30 border-emerald-400';
      case CELL_TYPES.PIT: return 'bg-red-500/30 border-red-400';
      case CELL_TYPES.START: return 'border-amber-400';
      default: return 'border-slate-600';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-mono">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
        
        .font-display { font-family: 'Space Grotesk', sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }
          50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.8); }
        }
        
        @keyframes sweep {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        
        .evaluating {
          background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
          background-size: 200% 100%;
          animation: sweep 1.5s ease-in-out infinite;
        }
        
        .improving {
          animation: pulse-glow 1s ease-in-out infinite;
        }
        
        .cell-transition {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .arrow-bounce {
          animation: bounce 0.5s ease-out;
        }
        
        @keyframes bounce {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.2); }
        }
      `}</style>
      
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="font-display text-4xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent mb-2">
            Policy Iteration
          </h1>
          <p className="text-slate-400 text-lg">
            Interactive visualization of the dynamic programming algorithm
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800 backdrop-blur">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                    phase === 'evaluation' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' :
                    phase === 'improvement' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50' :
                    phase === 'done' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50' :
                    'bg-slate-700/50 text-slate-400 border border-slate-600'
                  }`}>
                    {phase === 'evaluation' && `📊 Evaluating (sweep ${evalSweeps})`}
                    {phase === 'improvement' && '🔄 Improving Policy'}
                    {phase === 'done' && '✓ Optimal Policy Found'}
                    {phase === 'idle' && '⏸ Ready'}
                  </div>
                  <span className="text-slate-500">Iteration: {iteration}</span>
                </div>
              </div>

              <div className={`relative p-4 rounded-xl ${
                phase === 'evaluation' ? 'evaluating' : 
                phase === 'improvement' ? 'improving' : ''
              }`}>
                <div 
                  className="grid gap-2 mx-auto"
                  style={{ 
                    gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`,
                    maxWidth: '500px'
                  }}
                >
                  {grid.map((row, y) => 
                    row.map((cell, x) => (
                      <div
                        key={`${x}-${y}`}
                        className={`
                          cell-transition aspect-square rounded-xl border-2 
                          flex flex-col items-center justify-center relative
                          ${getCellStyle(cell)}
                          ${highlightedCell?.x === x && highlightedCell?.y === y ? 'ring-2 ring-cyan-400' : ''}
                        `}
                        style={{ 
                          backgroundColor: isTerminal(cell) ? undefined : getValueColor(values[y][x])
                        }}
                        onMouseEnter={() => setHighlightedCell({ x, y })}
                        onMouseLeave={() => setHighlightedCell(null)}
                      >
                        {cell === CELL_TYPES.GOAL && (
                          <span className="text-3xl">🎯</span>
                        )}
                        {cell === CELL_TYPES.PIT && (
                          <span className="text-3xl">🕳️</span>
                        )}
                        {cell === CELL_TYPES.START && (
                          <span className="absolute top-1 left-1 text-xs text-amber-400">START</span>
                        )}
                        
                        {!isTerminal(cell) && (
                          <>
                            <span className={`text-3xl mb-1 arrow-bounce`} key={policy[y][x]}>
                              {ACTIONS[policy[y][x]].arrow}
                            </span>
                            <span className="text-xs text-slate-300 font-semibold">
                              {values[y][x].toFixed(2)}
                            </span>
                          </>
                        )}
                        
                        {isTerminal(cell) && (
                          <span className="text-xs text-slate-300 font-semibold mt-1">
                            R: {getReward(cell)}
                          </span>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  disabled={policyStable}
                  className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                    policyStable 
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                      : isRunning 
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50 hover:bg-amber-500/30'
                        : 'bg-blue-500/20 text-blue-400 border border-blue-500/50 hover:bg-blue-500/30'
                  }`}
                >
                  {isRunning ? '⏸ Pause' : '▶ Auto Run'}
                </button>
                
                <button
                  onClick={step}
                  disabled={isRunning || policyStable}
                  className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                    isRunning || policyStable
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                      : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 hover:bg-cyan-500/30'
                  }`}
                >
                  ⏭ Step
                </button>
                
                <button
                  onClick={reset}
                  className="px-6 py-3 rounded-xl font-semibold bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-700 transition-all"
                >
                  ↺ Reset
                </button>

                <div className="flex items-center gap-2">
                  <span className="text-slate-500 text-sm">Speed:</span>
                  <input
                    type="range"
                    min="100"
                    max="1000"
                    step="100"
                    value={1100 - speed}
                    onChange={(e) => setSpeed(1100 - Number(e.target.value))}
                    className="w-24 accent-cyan-500"
                  />
                </div>
              </div>
            </div>

            <div className="mt-6 bg-slate-900/50 rounded-2xl p-6 border border-slate-800">
              <h3 className="font-display text-lg font-semibold text-slate-200 mb-4">Legend</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/30 border-2 border-emerald-400 flex items-center justify-center">🎯</div>
                  <span className="text-slate-400">Goal (+10)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-red-500/30 border-2 border-red-400 flex items-center justify-center">🕳️</div>
                  <span className="text-slate-400">Pit (-10)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg border-2 border-slate-600" style={{ backgroundColor: 'rgba(34, 197, 94, 0.5)' }}></div>
                  <span className="text-slate-400">High Value</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg border-2 border-slate-600" style={{ backgroundColor: 'rgba(239, 68, 68, 0.5)' }}></div>
                  <span className="text-slate-400">Low Value</span>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800">
              <h3 className="font-display text-lg font-semibold text-slate-200 mb-4">Algorithm</h3>
              <div className="space-y-4 text-sm">
                <div className={`p-4 rounded-xl border transition-all ${
                  phase === 'evaluation' 
                    ? 'bg-blue-500/10 border-blue-500/50 text-blue-300' 
                    : 'bg-slate-800/50 border-slate-700 text-slate-400'
                }`}>
                  <div className="font-semibold mb-2">1. Policy Evaluation</div>
                  <p>Compute value function V(s) for current policy π by iterating until convergence:</p>
                  <div className="mt-2 p-2 bg-slate-900/50 rounded font-mono text-xs">
                    V(s) ← R(s') + γ·V(s')
                  </div>
                </div>
                
                <div className={`p-4 rounded-xl border transition-all ${
                  phase === 'improvement' 
                    ? 'bg-purple-500/10 border-purple-500/50 text-purple-300' 
                    : 'bg-slate-800/50 border-slate-700 text-slate-400'
                }`}>
                  <div className="font-semibold mb-2">2. Policy Improvement</div>
                  <p>Update policy to be greedy with respect to value function:</p>
                  <div className="mt-2 p-2 bg-slate-900/50 rounded font-mono text-xs">
                    π(s) ← argmax_a [R + γ·V(s')]
                  </div>
                </div>
                
                <div className={`p-4 rounded-xl border transition-all ${
                  phase === 'done' 
                    ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-300' 
                    : 'bg-slate-800/50 border-slate-700 text-slate-400'
                }`}>
                  <div className="font-semibold mb-2">3. Check Convergence</div>
                  <p>If policy unchanged → optimal! Otherwise, repeat from step 1.</p>
                </div>
              </div>
            </div>

            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800">
              <h3 className="font-display text-lg font-semibold text-slate-200 mb-4">Parameters</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Discount (γ):</span>
                  <span className="text-cyan-400 font-semibold">{GAMMA}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Threshold (θ):</span>
                  <span className="text-cyan-400 font-semibold">{THETA}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Grid Size:</span>
                  <span className="text-cyan-400 font-semibold">{GRID_SIZE}×{GRID_SIZE}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Step Cost:</span>
                  <span className="text-cyan-400 font-semibold">-0.1</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-800 max-h-64 overflow-y-auto">
              <h3 className="font-display text-lg font-semibold text-slate-200 mb-4">Activity Log</h3>
              <div className="space-y-2 text-xs">
                {history.length === 0 && (
                  <p className="text-slate-500 italic">Press "Auto Run" or "Step" to begin...</p>
                )}
                {history.slice().reverse().map((entry, i) => (
                  <div key={i} className={`p-2 rounded-lg ${
                    entry.type === 'done' ? 'bg-emerald-500/10 text-emerald-400' :
                    entry.type === 'improved' ? 'bg-purple-500/10 text-purple-400' :
                    entry.type === 'eval_done' ? 'bg-blue-500/10 text-blue-400' :
                    'bg-slate-800/50 text-slate-400'
                  }`}>
                    {entry.type === 'start_eval' && `▶ Starting iteration ${entry.iteration}`}
                    {entry.type === 'eval_done' && `✓ Evaluation converged in ${entry.sweeps} sweeps`}
                    {entry.type === 'improved' && '↻ Policy improved, continuing...'}
                    {entry.type === 'done' && '★ Optimal policy found!'}
                  </div>
                ))}
              </div>
            </div>

            {highlightedCell && (
              <div className="bg-slate-900/50 rounded-2xl p-6 border border-cyan-500/30">
                <h3 className="font-display text-lg font-semibold text-cyan-400 mb-3">
                  Cell ({highlightedCell.x}, {highlightedCell.y})
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Type:</span>
                    <span className="text-slate-200 capitalize">{grid[highlightedCell.y][highlightedCell.x]}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Value:</span>
                    <span className="text-slate-200">{values[highlightedCell.y][highlightedCell.x].toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Policy:</span>
                    <span className="text-slate-200">{ACTIONS[policy[highlightedCell.y][highlightedCell.x]].name} {ACTIONS[policy[highlightedCell.y][highlightedCell.x]].arrow}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="mt-8 text-center text-slate-500 text-sm">
          <p>Hover over cells to see details • γ = {GAMMA} (discount factor) • θ = {THETA} (convergence threshold)</p>
        </footer>
      </div>
    </div>
  );
}
