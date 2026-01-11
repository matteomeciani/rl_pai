import React, { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 4;
const ACTIONS = ['↑', '→', '↓', '←'];
const ACTION_DELTAS = { '↑': [-1, 0], '→': [0, 1], '↓': [1, 0], '←': [0, -1] };

const initialQTable = () => {
  const q = {};
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      q[`${r},${c}`] = { '↑': 0, '→': 0, '↓': 0, '←': 0 };
    }
  }
  return q;
};

const getReward = (row, col) => {
  if (row === 0 && col === GRID_SIZE - 1) return 10; // Goal
  if (row === 1 && (col === 1 || col === 2)) return -5; // Danger
  return -0.1; // Step cost
};

const isGoal = (row, col) => row === 0 && col === GRID_SIZE - 1;
const isDanger = (row, col) => row === 1 && (col === 1 || col === 2);

const clamp = (val, min, max) => Math.max(min, Math.min(max, val));

const getNextState = (row, col, action) => {
  const [dr, dc] = ACTION_DELTAS[action];
  return [clamp(row + dr, 0, GRID_SIZE - 1), clamp(col + dc, 0, GRID_SIZE - 1)];
};

const getMaxAction = (qTable, state) => {
  const actions = qTable[state];
  let maxA = '↑', maxV = -Infinity;
  for (const a of ACTIONS) {
    if (actions[a] > maxV) { maxV = actions[a]; maxA = a; }
  }
  return maxA;
};

const getEpsilonGreedyAction = (qTable, state, epsilon = 0.3) => {
  if (Math.random() < epsilon) {
    return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
  }
  return getMaxAction(qTable, state);
};

export default function RLVisualization() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(800);
  const [episode, setEpisode] = useState(1);
  
  // Separate states for each algorithm
  const [sarsaState, setSarsaState] = useState({ row: GRID_SIZE - 1, col: 0 });
  const [qLearningState, setQLearningState] = useState({ row: GRID_SIZE - 1, col: 0 });
  
  const [sarsaQ, setSarsaQ] = useState(initialQTable);
  const [qLearningQ, setQLearningQ] = useState(initialQTable);
  
  const [sarsaHighlight, setSarsaHighlight] = useState(null);
  const [qLearningHighlight, setQLearningHighlight] = useState(null);
  
  const [sarsaAction, setSarsaAction] = useState(null);
  const [sarsaNextAction, setSarsaNextAction] = useState(null);
  const [qLearningAction, setQLearningAction] = useState(null);
  const [qLearningMaxAction, setQLearningMaxAction] = useState(null);
  
  const [updatePhase, setUpdatePhase] = useState('idle'); // idle, action, nextState, update
  const [sarsaUpdateInfo, setSarsaUpdateInfo] = useState(null);
  const [qLearningUpdateInfo, setQLearningUpdateInfo] = useState(null);

  const alpha = 0.5;
  const gamma = 0.9;
  const epsilon = 0.3;

  const resetEpisode = useCallback(() => {
    setSarsaState({ row: GRID_SIZE - 1, col: 0 });
    setQLearningState({ row: GRID_SIZE - 1, col: 0 });
    setSarsaAction(null);
    setSarsaNextAction(null);
    setQLearningAction(null);
    setQLearningMaxAction(null);
    setSarsaHighlight(null);
    setQLearningHighlight(null);
    setUpdatePhase('idle');
    setSarsaUpdateInfo(null);
    setQLearningUpdateInfo(null);
  }, []);

  const performStep = useCallback(() => {
    const sState = `${sarsaState.row},${sarsaState.col}`;
    const qState = `${qLearningState.row},${qLearningState.col}`;
    
    // Check if either agent reached goal
    if (isGoal(sarsaState.row, sarsaState.col) && isGoal(qLearningState.row, qLearningState.col)) {
      setEpisode(e => e + 1);
      resetEpisode();
      return;
    }

    if (updatePhase === 'idle') {
      // Phase 1: Choose actions
      const sAction = getEpsilonGreedyAction(sarsaQ, sState, epsilon);
      const qAction = getEpsilonGreedyAction(qLearningQ, qState, epsilon);
      setSarsaAction(sAction);
      setQLearningAction(qAction);
      setUpdatePhase('action');
      
    } else if (updatePhase === 'action') {
      // Phase 2: Execute actions, observe next state
      const [sNextRow, sNextCol] = getNextState(sarsaState.row, sarsaState.col, sarsaAction);
      const [qNextRow, qNextCol] = getNextState(qLearningState.row, qLearningState.col, qLearningAction);
      
      const sNextState = `${sNextRow},${sNextCol}`;
      const qNextState = `${qNextRow},${qNextCol}`;
      
      // SARSA: choose next action (ON-POLICY - this is what we'll actually do!)
      const sNextAction = getEpsilonGreedyAction(sarsaQ, sNextState, epsilon);
      setSarsaNextAction(sNextAction);
      
      // Q-Learning: find max action (OFF-POLICY - we won't necessarily do this!)
      const qMaxAction = getMaxAction(qLearningQ, qNextState);
      setQLearningMaxAction(qMaxAction);
      
      setSarsaHighlight({ row: sNextRow, col: sNextCol });
      setQLearningHighlight({ row: qNextRow, col: qNextCol });
      
      // Prepare update info
      const sReward = getReward(sNextRow, sNextCol);
      const qReward = getReward(qNextRow, qNextCol);
      
      setSarsaUpdateInfo({
        state: sState,
        action: sarsaAction,
        reward: sReward,
        nextState: sNextState,
        nextAction: sNextAction,
        nextQ: sarsaQ[sNextState][sNextAction],
        oldQ: sarsaQ[sState][sarsaAction]
      });
      
      setQLearningUpdateInfo({
        state: qState,
        action: qLearningAction,
        reward: qReward,
        nextState: qNextState,
        maxAction: qMaxAction,
        maxQ: qLearningQ[qNextState][qMaxAction],
        oldQ: qLearningQ[qState][qLearningAction]
      });
      
      setUpdatePhase('nextState');
      
    } else if (updatePhase === 'nextState') {
      // Phase 3: Update Q-values
      if (sarsaUpdateInfo && qLearningUpdateInfo) {
        // SARSA Update
        setSarsaQ(prev => {
          const newQ = JSON.parse(JSON.stringify(prev));
          const { state, action, reward, nextQ, oldQ } = sarsaUpdateInfo;
          newQ[state][action] = oldQ + alpha * (reward + gamma * nextQ - oldQ);
          return newQ;
        });
        
        // Q-Learning Update
        setQLearningQ(prev => {
          const newQ = JSON.parse(JSON.stringify(prev));
          const { state, action, reward, maxQ, oldQ } = qLearningUpdateInfo;
          newQ[state][action] = oldQ + alpha * (reward + gamma * maxQ - oldQ);
          return newQ;
        });
      }
      
      setUpdatePhase('update');
      
    } else if (updatePhase === 'update') {
      // Phase 4: Move to next state
      const [sNextRow, sNextCol] = getNextState(sarsaState.row, sarsaState.col, sarsaAction);
      const [qNextRow, qNextCol] = getNextState(qLearningState.row, qLearningState.col, qLearningAction);
      
      setSarsaState({ row: sNextRow, col: sNextCol });
      setQLearningState({ row: qNextRow, col: qNextCol });
      
      setSarsaAction(null);
      setSarsaNextAction(null);
      setQLearningAction(null);
      setQLearningMaxAction(null);
      setSarsaHighlight(null);
      setQLearningHighlight(null);
      setSarsaUpdateInfo(null);
      setQLearningUpdateInfo(null);
      setUpdatePhase('idle');
      setStep(s => s + 1);
    }
  }, [sarsaState, qLearningState, sarsaQ, qLearningQ, sarsaAction, qLearningAction, 
      sarsaNextAction, updatePhase, sarsaUpdateInfo, qLearningUpdateInfo, epsilon, resetEpisode]);

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(performStep, speed);
      return () => clearInterval(interval);
    }
  }, [isPlaying, performStep, speed]);

  const reset = () => {
    setStep(0);
    setEpisode(1);
    setIsPlaying(false);
    setSarsaQ(initialQTable());
    setQLearningQ(initialQTable());
    resetEpisode();
  };

  const GridCell = ({ row, col, qTable, agentPos, highlight, algorithm, selectedAction, nextAction, maxAction }) => {
    const isAgent = agentPos.row === row && agentPos.col === col;
    const isHighlighted = highlight && highlight.row === row && highlight.col === col;
    const isGoalCell = isGoal(row, col);
    const isDangerCell = isDanger(row, col);
    const isStart = row === GRID_SIZE - 1 && col === 0;
    
    const state = `${row},${col}`;
    const qValues = qTable[state];
    const bestAction = getMaxAction(qTable, state);
    
    return (
      <div className={`
        relative w-24 h-24 border-2 transition-all duration-300
        ${isGoalCell ? 'bg-emerald-900/60 border-emerald-400' : 
          isDangerCell ? 'bg-red-900/60 border-red-400' : 
          isStart ? 'bg-blue-900/40 border-blue-400' :
          'bg-slate-800/60 border-slate-600'}
        ${isHighlighted ? 'ring-4 ring-yellow-400 ring-opacity-80 scale-105' : ''}
        ${isAgent ? 'ring-4 ring-cyan-400' : ''}
      `}>
        {/* Q-value arrows */}
        <div className="absolute inset-0 flex items-center justify-center">
          {ACTIONS.map((action, i) => {
            const isSelected = selectedAction === action && isAgent;
            const isNext = algorithm === 'sarsa' && nextAction === action && isHighlighted;
            const isMax = algorithm === 'qlearning' && maxAction === action && isHighlighted;
            const isBest = bestAction === action;
            
            const positions = {
              '↑': 'top-1 left-1/2 -translate-x-1/2',
              '→': 'right-1 top-1/2 -translate-y-1/2',
              '↓': 'bottom-1 left-1/2 -translate-x-1/2',
              '←': 'left-1 top-1/2 -translate-y-1/2'
            };
            
            return (
              <div key={action} className={`
                absolute ${positions[action]} text-xs font-mono px-1.5 py-0.5 rounded
                transition-all duration-300
                ${isSelected ? 'bg-cyan-500 text-white scale-125 font-bold' : ''}
                ${isNext ? 'bg-orange-500 text-white scale-125 font-bold animate-pulse' : ''}
                ${isMax ? 'bg-purple-500 text-white scale-125 font-bold animate-pulse' : ''}
                ${!isSelected && !isNext && !isMax && isBest ? 'text-emerald-400 font-semibold' : ''}
                ${!isSelected && !isNext && !isMax && !isBest ? 'text-slate-400' : ''}
              `}>
                {qValues[action].toFixed(1)}
              </div>
            );
          })}
        </div>
        
        {/* Cell labels */}
        {isGoalCell && <div className="absolute inset-0 flex items-center justify-center text-2xl">🎯</div>}
        {isDangerCell && <div className="absolute inset-0 flex items-center justify-center text-2xl">⚡</div>}
        {isStart && !isAgent && <div className="absolute inset-0 flex items-center justify-center text-lg text-blue-300">START</div>}
        
        {/* Agent */}
        {isAgent && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className={`text-3xl transform transition-transform duration-200 ${selectedAction ? 'scale-110' : ''}`}>
              🤖
            </div>
          </div>
        )}
      </div>
    );
  };

  const UpdateFormula = ({ info, algorithm }) => {
    if (!info) return null;
    
    const isSarsa = algorithm === 'sarsa';
    const targetValue = isSarsa ? info.nextQ : info.maxQ;
    const targetLabel = isSarsa ? `Q(s', a')` : `max Q(s', a')`;
    const newQ = info.oldQ + alpha * (info.reward + gamma * targetValue - info.oldQ);
    
    return (
      <div className={`
        p-4 rounded-xl border-2 transition-all duration-500
        ${updatePhase === 'update' ? 'scale-105' : ''}
        ${isSarsa ? 'bg-orange-950/50 border-orange-500/50' : 'bg-purple-950/50 border-purple-500/50'}
      `}>
        <div className="font-mono text-sm space-y-2">
          <div className="text-slate-300">
            Q({info.state}, {info.action}) ← {info.oldQ.toFixed(2)} + {alpha} × [
          </div>
          <div className="pl-4 flex items-center gap-2">
            <span className="text-yellow-400">{info.reward.toFixed(1)}</span>
            <span className="text-slate-400">+</span>
            <span className="text-slate-400">{gamma} ×</span>
            <span className={`px-2 py-1 rounded font-bold ${isSarsa ? 'bg-orange-600 text-white' : 'bg-purple-600 text-white'}`}>
              {targetLabel} = {targetValue.toFixed(2)}
            </span>
            <span className="text-slate-400">- {info.oldQ.toFixed(2)}</span>
            <span className="text-slate-300">]</span>
          </div>
          <div className="text-emerald-400 font-bold pt-2 border-t border-slate-600">
            = {newQ.toFixed(3)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { font-family: 'Space Grotesk', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
      `}</style>
      
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-orange-400 bg-clip-text text-transparent mb-2">
          On-Policy vs Off-Policy Learning
        </h1>
        <p className="text-slate-400 text-lg">
          See exactly WHY SARSA is on-policy and Q-Learning is off-policy
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-8 flex-wrap">
        <button onClick={() => setIsPlaying(!isPlaying)}
          className={`px-6 py-3 rounded-xl font-semibold transition-all ${
            isPlaying ? 'bg-red-600 hover:bg-red-500' : 'bg-emerald-600 hover:bg-emerald-500'
          }`}>
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button onClick={performStep} disabled={isPlaying}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold transition-all disabled:opacity-50">
          ⏭ Step
        </button>
        <button onClick={reset}
          className="px-6 py-3 bg-slate-600 hover:bg-slate-500 rounded-xl font-semibold transition-all">
          🔄 Reset
        </button>
        <div className="flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-xl">
          <span className="text-sm text-slate-400">Speed:</span>
          <input type="range" min="200" max="1500" value={1700 - speed} 
            onChange={e => setSpeed(1700 - e.target.value)}
            className="w-24" />
        </div>
        <div className="flex items-center gap-4 bg-slate-800 px-4 py-2 rounded-xl">
          <span className="text-sm">Episode: <span className="text-cyan-400 font-bold">{episode}</span></span>
          <span className="text-sm">Step: <span className="text-cyan-400 font-bold">{step}</span></span>
        </div>
      </div>

      {/* Phase Indicator */}
      <div className="flex justify-center mb-6">
        <div className="flex gap-2 bg-slate-800/50 p-2 rounded-xl">
          {['idle', 'action', 'nextState', 'update'].map((phase, i) => (
            <div key={phase} className={`
              px-4 py-2 rounded-lg text-sm font-medium transition-all
              ${updatePhase === phase ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400'}
            `}>
              {i + 1}. {phase === 'idle' ? 'Ready' : phase === 'action' ? 'Choose Action' : 
                phase === 'nextState' ? 'Observe Next' : 'Update Q'}
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 max-w-7xl mx-auto">
        
        {/* SARSA Panel */}
        <div className="bg-slate-800/30 rounded-2xl p-6 border-2 border-orange-500/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-4 h-4 rounded-full bg-orange-500 animate-pulse"></div>
            <h2 className="text-2xl font-bold text-orange-400">SARSA</h2>
            <span className="px-3 py-1 bg-orange-600/30 text-orange-300 rounded-full text-sm font-semibold">
              ON-POLICY
            </span>
          </div>
          
          <div className="bg-slate-900/50 rounded-xl p-4 mb-4">
            <p className="text-slate-300 text-sm leading-relaxed">
              Uses the <span className="text-orange-400 font-bold">actual next action a'</span> that will be taken 
              (sampled from the same ε-greedy policy). Learns Q<sup>π</sup> — the value of the policy being followed.
            </p>
          </div>

          <div className="flex justify-center mb-4">
            <div className="grid grid-cols-4 gap-1">
              {Array.from({ length: GRID_SIZE }, (_, r) =>
                Array.from({ length: GRID_SIZE }, (_, c) => (
                  <GridCell key={`sarsa-${r}-${c}`} row={r} col={c} 
                    qTable={sarsaQ} agentPos={sarsaState} highlight={sarsaHighlight}
                    algorithm="sarsa" selectedAction={sarsaAction} 
                    nextAction={sarsaNextAction} maxAction={null} />
                ))
              )}
            </div>
          </div>

          {/* SARSA Key Insight */}
          <div className={`
            p-4 rounded-xl border-2 mb-4 transition-all duration-300
            ${updatePhase === 'nextState' || updatePhase === 'update' ? 
              'bg-orange-900/40 border-orange-400 scale-102' : 'bg-slate-800/50 border-slate-600'}
          `}>
            <div className="text-sm font-semibold text-orange-300 mb-2">🔑 THE KEY DIFFERENCE:</div>
            <div className="text-slate-200">
              Bootstrap target uses <span className="bg-orange-600 px-2 py-1 rounded font-mono font-bold">Q(s', a')</span>
              <br />
              <span className="text-slate-400 text-sm">where a' is the action we WILL actually take next</span>
            </div>
          </div>

          <UpdateFormula info={sarsaUpdateInfo} algorithm="sarsa" />
        </div>

        {/* Q-Learning Panel */}
        <div className="bg-slate-800/30 rounded-2xl p-6 border-2 border-purple-500/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-4 h-4 rounded-full bg-purple-500 animate-pulse"></div>
            <h2 className="text-2xl font-bold text-purple-400">Q-Learning</h2>
            <span className="px-3 py-1 bg-purple-600/30 text-purple-300 rounded-full text-sm font-semibold">
              OFF-POLICY
            </span>
          </div>
          
          <div className="bg-slate-900/50 rounded-xl p-4 mb-4">
            <p className="text-slate-300 text-sm leading-relaxed">
              Uses <span className="text-purple-400 font-bold">max over all actions</span> regardless of what 
              action will actually be taken. Learns Q* — the optimal value function.
            </p>
          </div>

          <div className="flex justify-center mb-4">
            <div className="grid grid-cols-4 gap-1">
              {Array.from({ length: GRID_SIZE }, (_, r) =>
                Array.from({ length: GRID_SIZE }, (_, c) => (
                  <GridCell key={`ql-${r}-${c}`} row={r} col={c} 
                    qTable={qLearningQ} agentPos={qLearningState} highlight={qLearningHighlight}
                    algorithm="qlearning" selectedAction={qLearningAction} 
                    nextAction={null} maxAction={qLearningMaxAction} />
                ))
              )}
            </div>
          </div>

          {/* Q-Learning Key Insight */}
          <div className={`
            p-4 rounded-xl border-2 mb-4 transition-all duration-300
            ${updatePhase === 'nextState' || updatePhase === 'update' ? 
              'bg-purple-900/40 border-purple-400 scale-102' : 'bg-slate-800/50 border-slate-600'}
          `}>
            <div className="text-sm font-semibold text-purple-300 mb-2">🔑 THE KEY DIFFERENCE:</div>
            <div className="text-slate-200">
              Bootstrap target uses <span className="bg-purple-600 px-2 py-1 rounded font-mono font-bold">max Q(s', a)</span>
              <br />
              <span className="text-slate-400 text-sm">regardless of what action we'll actually take</span>
            </div>
          </div>

          <UpdateFormula info={qLearningUpdateInfo} algorithm="qlearning" />
        </div>
      </div>

      {/* Legend & Explanation */}
      <div className="max-w-4xl mx-auto mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 rounded-xl p-5">
          <h3 className="font-bold text-lg mb-3 text-slate-200">Legend</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🤖</span>
              <span className="text-slate-300">Agent</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              <span className="text-slate-300">Goal (+10)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-2xl">⚡</span>
              <span className="text-slate-300">Danger (-5)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 bg-cyan-500 rounded"></div>
              <span className="text-slate-300">Current action</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 bg-orange-500 rounded"></div>
              <span className="text-slate-300">SARSA: a' (next action)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 bg-purple-500 rounded"></div>
              <span className="text-slate-300">Q-L: max action</span>
            </div>
          </div>
        </div>
        
        <div className="bg-slate-800/50 rounded-xl p-5">
          <h3 className="font-bold text-lg mb-3 text-slate-200">Why This Matters</h3>
          <div className="text-sm text-slate-300 space-y-2">
            <p><span className="text-orange-400 font-semibold">SARSA</span> accounts for exploration mistakes — it learns the value <em>including</em> the risk of ε-greedy exploration.</p>
            <p><span className="text-purple-400 font-semibold">Q-Learning</span> assumes optimal future behavior — it can learn from any data but may underestimate danger.</p>
          </div>
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="max-w-xl mx-auto mt-6 bg-slate-800/30 rounded-xl p-4">
        <div className="flex justify-center gap-8 text-sm mono">
          <span>α = {alpha}</span>
          <span>γ = {gamma}</span>
          <span>ε = {epsilon}</span>
        </div>
      </div>
    </div>
  );
}
