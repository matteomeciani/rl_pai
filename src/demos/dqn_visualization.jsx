import React, { useState, useEffect, useCallback } from 'react';

const DQNVisualization = () => {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1500);
  const [replayBuffer, setReplayBuffer] = useState([]);
  const [onlineWeights, setOnlineWeights] = useState([0.5, 0.3, 0.7, 0.4, 0.6, 0.5]);
  const [targetWeights, setTargetWeights] = useState([0.5, 0.3, 0.7, 0.4, 0.6, 0.5]);
  const [currentPhase, setCurrentPhase] = useState('collect');
  const [syncCounter, setSyncCounter] = useState(0);
  const [highlightedSample, setHighlightedSample] = useState(null);
  const [showTarget, setShowTarget] = useState(false);
  const [showLoss, setShowLoss] = useState(false);
  const [lossValue, setLossValue] = useState(null);
  const [qPrediction, setQPrediction] = useState(null);
  const [targetValue, setTargetValue] = useState(null);
  const [justSynced, setJustSynced] = useState(false);
  const [agentPosition, setAgentPosition] = useState(0);

  const SYNC_INTERVAL = 4;
  const BUFFER_MAX = 8;

  const phases = ['collect', 'store', 'sample', 'compute_target', 'compute_loss', 'update', 'check_sync'];
  const phaseDescriptions = {
    collect: 'Agent interacts with environment, collecting experience (s, a, r, s\')',
    store: 'Store transition in replay buffer',
    sample: 'Sample random minibatch from replay buffer',
    compute_target: 'Compute target using FROZEN target network: y = r + γ·max Q(s\',a\';θ⁻)',
    compute_loss: 'Compute TD loss: L = (y - Q(s,a;θ))²',
    update: 'Update online network weights via gradient descent',
    check_sync: syncCounter >= SYNC_INTERVAL - 1 ? '⚡ SYNC! Copy online weights to target network' : `Steps until sync: ${SYNC_INTERVAL - syncCounter - 1}`
  };

  const generateExperience = useCallback(() => {
    const states = ['s₁', 's₂', 's₃', 's₄', 's₅'];
    const actions = ['←', '→', '↑', '↓'];
    const s = states[Math.floor(Math.random() * states.length)];
    const a = actions[Math.floor(Math.random() * actions.length)];
    const r = (Math.random() * 2 - 0.5).toFixed(1);
    const s_next = states[Math.floor(Math.random() * states.length)];
    return { s, a, r, s_next, id: Date.now() + Math.random() };
  }, []);

  const advancePhase = useCallback(() => {
    setStep(prev => prev + 1);
    
    setCurrentPhase(prev => {
      const currentIndex = phases.indexOf(prev);
      const nextIndex = (currentIndex + 1) % phases.length;
      const nextPhase = phases[nextIndex];

      if (nextPhase === 'collect') {
        setAgentPosition(p => (p + 1) % 5);
        setShowTarget(false);
        setShowLoss(false);
        setHighlightedSample(null);
        setLossValue(null);
        setQPrediction(null);
        setTargetValue(null);
      }

      if (nextPhase === 'store') {
        const newExp = generateExperience();
        setReplayBuffer(buf => {
          const newBuf = [...buf, newExp];
          if (newBuf.length > BUFFER_MAX) newBuf.shift();
          return newBuf;
        });
      }

      if (nextPhase === 'sample' && replayBuffer.length > 0) {
        const randomIndex = Math.floor(Math.random() * replayBuffer.length);
        setHighlightedSample(replayBuffer[randomIndex]?.id);
      }

      if (nextPhase === 'compute_target') {
        setShowTarget(true);
        const target = (Math.random() * 10 + 5).toFixed(2);
        setTargetValue(target);
      }

      if (nextPhase === 'compute_loss') {
        setShowLoss(true);
        const pred = (Math.random() * 10 + 3).toFixed(2);
        setQPrediction(pred);
        if (targetValue) {
          const loss = Math.pow(parseFloat(targetValue) - parseFloat(pred), 2).toFixed(3);
          setLossValue(loss);
        }
      }

      if (nextPhase === 'update') {
        setOnlineWeights(prev => prev.map(w => {
          const delta = (Math.random() - 0.5) * 0.1;
          return Math.max(0, Math.min(1, w + delta));
        }));
      }

      if (nextPhase === 'check_sync') {
        setSyncCounter(prev => {
          const newCount = prev + 1;
          if (newCount >= SYNC_INTERVAL) {
            setTargetWeights([...onlineWeights]);
            setJustSynced(true);
            setTimeout(() => setJustSynced(false), 800);
            return 0;
          }
          return newCount;
        });
      }

      return nextPhase;
    });
  }, [generateExperience, replayBuffer, onlineWeights, targetValue, phases]);

  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(advancePhase, speed);
    return () => clearInterval(interval);
  }, [isPlaying, speed, advancePhase]);

  const reset = () => {
    setStep(0);
    setIsPlaying(false);
    setReplayBuffer([]);
    setOnlineWeights([0.5, 0.3, 0.7, 0.4, 0.6, 0.5]);
    setTargetWeights([0.5, 0.3, 0.7, 0.4, 0.6, 0.5]);
    setCurrentPhase('collect');
    setSyncCounter(0);
    setHighlightedSample(null);
    setShowTarget(false);
    setShowLoss(false);
    setLossValue(null);
    setQPrediction(null);
    setTargetValue(null);
    setAgentPosition(0);
  };

  const NeuralNetwork = ({ weights, label, isTarget, isActive, isSyncing }) => (
    <div className={`relative p-4 rounded-2xl border-2 transition-all duration-500 ${
      isSyncing ? 'border-yellow-400 bg-yellow-400/20 scale-105' :
      isTarget ? 'border-cyan-500/50 bg-gradient-to-br from-slate-900 to-cyan-950' :
      'border-emerald-500/50 bg-gradient-to-br from-slate-900 to-emerald-950'
    } ${isActive ? 'ring-2 ring-white/50' : ''}`}>
      
      <div className={`text-xs font-bold mb-3 tracking-wider ${isTarget ? 'text-cyan-400' : 'text-emerald-400'}`}>
        {label}
      </div>
      
      {isTarget && (
        <div className="absolute -top-2 -right-2 bg-cyan-500 text-black text-xs px-2 py-0.5 rounded-full font-bold">
          FROZEN
        </div>
      )}
      
      <svg viewBox="0 0 120 100" className="w-full h-32">
        {[0, 1, 2].map(i => (
          <circle key={`i${i}`} cx="20" cy={25 + i * 25} r="8" 
            className={`${isTarget ? 'fill-cyan-600' : 'fill-emerald-600'} transition-all`}
            style={{ opacity: 0.5 + weights[i] * 0.5 }}
          />
        ))}
        
        {[0, 1, 2, 3].map(i => (
          <circle key={`h${i}`} cx="60" cy={15 + i * 23} r="8"
            className={`${isTarget ? 'fill-cyan-500' : 'fill-emerald-500'} transition-all`}
            style={{ opacity: 0.5 + weights[(i + 2) % 6] * 0.5 }}
          />
        ))}
        
        {[0, 1].map(i => (
          <circle key={`o${i}`} cx="100" cy={35 + i * 30} r="8"
            className={`${isTarget ? 'fill-cyan-400' : 'fill-emerald-400'} transition-all`}
            style={{ opacity: 0.5 + weights[(i + 4) % 6] * 0.5 }}
          />
        ))}
        
        {[0, 1, 2].map(i => 
          [0, 1, 2, 3].map(j => (
            <line key={`ih${i}${j}`} x1="28" y1={25 + i * 25} x2="52" y2={15 + j * 23}
              className={`${isTarget ? 'stroke-cyan-700' : 'stroke-emerald-700'}`}
              strokeWidth={0.5 + weights[(i + j) % 6] * 1.5}
              opacity={0.3 + weights[(i + j) % 6] * 0.4}
            />
          ))
        )}
        
        {[0, 1, 2, 3].map(i => 
          [0, 1].map(j => (
            <line key={`ho${i}${j}`} x1="68" y1={15 + i * 23} x2="92" y2={35 + j * 30}
              className={`${isTarget ? 'stroke-cyan-700' : 'stroke-emerald-700'}`}
              strokeWidth={0.5 + weights[(i + j + 2) % 6] * 1.5}
              opacity={0.3 + weights[(i + j + 2) % 6] * 0.4}
            />
          ))
        )}
      </svg>
      
      <div className={`text-xs mt-2 font-mono ${isTarget ? 'text-cyan-300' : 'text-emerald-300'}`}>
        θ{isTarget ? '⁻' : ''} = [{weights.map(w => w.toFixed(2)).join(', ')}]
      </div>
    </div>
  );

  const gridPositions = [
    { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 },
    { x: 1, y: 1 }, { x: 2, y: 1 }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 font-sans">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');
        * { font-family: 'Space Grotesk', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 5px currentColor; }
          50% { box-shadow: 0 0 20px currentColor, 0 0 30px currentColor; }
        }
        .pulse-glow { animation: pulse-glow 1s ease-in-out infinite; }
        @keyframes flow {
          0% { stroke-dashoffset: 20; }
          100% { stroke-dashoffset: 0; }
        }
        .flow-line { stroke-dasharray: 5 5; animation: flow 0.5s linear infinite; }
        @keyframes bounce-in {
          0% { transform: scale(0); }
          50% { transform: scale(1.2); }
          100% { transform: scale(1); }
        }
        .bounce-in { animation: bounce-in 0.3s ease-out; }
      `}</style>

      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
            DQN: Two Networks Learning Together
          </h1>
          <p className="text-slate-400 mt-2">Understanding the Online Network and Target Network</p>
        </div>

        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              isPlaying 
                ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30' 
                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-500/30'
            }`}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            onClick={advancePhase}
            disabled={isPlaying}
            className="px-6 py-2 rounded-lg font-semibold bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Step →
          </button>
          <button
            onClick={reset}
            className="px-6 py-2 rounded-lg font-semibold bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-600/50"
          >
            ↺ Reset
          </button>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="px-4 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-300"
          >
            <option value={2500}>Slow</option>
            <option value={1500}>Medium</option>
            <option value={800}>Fast</option>
          </select>
        </div>

        <div className={`mb-6 p-4 rounded-xl border-2 transition-all duration-300 ${
          currentPhase === 'check_sync' && syncCounter >= SYNC_INTERVAL - 1
            ? 'border-yellow-500 bg-yellow-500/10'
            : 'border-slate-700 bg-slate-900/50'
        }`}>
          <div className="flex items-center gap-4">
            <div className="text-sm text-slate-500">Step {step}</div>
            <div className="flex-1">
              <div className="flex gap-1 mb-2">
                {phases.map((phase, i) => (
                  <div
                    key={phase}
                    className={`flex-1 h-2 rounded-full transition-all ${
                      phase === currentPhase 
                        ? 'bg-emerald-500' 
                        : phases.indexOf(currentPhase) > i 
                          ? 'bg-emerald-800' 
                          : 'bg-slate-700'
                    }`}
                  />
                ))}
              </div>
              <div className="text-emerald-400 font-semibold">
                {currentPhase.replace('_', ' ').toUpperCase()}
              </div>
              <div className="text-slate-400 text-sm mt-1">
                {phaseDescriptions[currentPhase]}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">Sync Progress</div>
              <div className="flex gap-1 mt-1">
                {[...Array(SYNC_INTERVAL)].map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-full transition-all ${
                      i < syncCounter ? 'bg-yellow-500' : 'bg-slate-700'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3">
            <div className={`p-4 rounded-xl border-2 transition-all ${
              currentPhase === 'collect' ? 'border-purple-500 bg-purple-500/10' : 'border-slate-700 bg-slate-900/50'
            }`}>
              <div className="text-xs text-purple-400 font-bold mb-3 tracking-wider">ENVIRONMENT</div>
              <div className="grid grid-cols-3 gap-1 mb-3">
                {gridPositions.map((pos, i) => (
                  <div
                    key={i}
                    className={`aspect-square rounded flex items-center justify-center text-lg transition-all ${
                      i === agentPosition 
                        ? 'bg-purple-500 text-white scale-110' 
                        : 'bg-slate-800 text-slate-600'
                    }`}
                    style={{ gridColumn: pos.x + 1, gridRow: pos.y + 1 }}
                  >
                    {i === agentPosition ? '🤖' : '·'}
                  </div>
                ))}
              </div>
              <div className="text-xs text-slate-500 text-center">
                Agent at s{agentPosition + 1}
              </div>
            </div>

            <div className={`mt-4 p-4 rounded-xl border-2 transition-all ${
              currentPhase === 'store' || currentPhase === 'sample' 
                ? 'border-amber-500 bg-amber-500/10' 
                : 'border-slate-700 bg-slate-900/50'
            }`}>
              <div className="text-xs text-amber-400 font-bold mb-3 tracking-wider">
                REPLAY BUFFER ({replayBuffer.length}/{BUFFER_MAX})
              </div>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {replayBuffer.length === 0 ? (
                  <div className="text-slate-600 text-xs text-center py-4">Empty</div>
                ) : (
                  replayBuffer.map((exp, i) => (
                    <div
                      key={exp.id}
                      className={`text-xs p-2 rounded font-mono transition-all ${
                        highlightedSample === exp.id
                          ? 'bg-amber-500 text-black scale-105 pulse-glow'
                          : 'bg-slate-800 text-slate-400'
                      } ${currentPhase === 'store' && i === replayBuffer.length - 1 ? 'bounce-in' : ''}`}
                    >
                      ({exp.s}, {exp.a}, {exp.r}, {exp.s_next})
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="col-span-6">
            <div className="grid grid-cols-2 gap-4">
              <NeuralNetwork
                weights={onlineWeights}
                label="ONLINE NETWORK Q(s,a;θ)"
                isTarget={false}
                isActive={currentPhase === 'update' || currentPhase === 'compute_loss'}
                isSyncing={false}
              />
              <NeuralNetwork
                weights={targetWeights}
                label="TARGET NETWORK Q(s,a;θ⁻)"
                isTarget={true}
                isActive={currentPhase === 'compute_target'}
                isSyncing={justSynced}
              />
            </div>

            <svg className="w-full h-16 -mt-2" viewBox="0 0 400 60">
              {justSynced && (
                <>
                  <line x1="200" y1="30" x2="300" y2="30" 
                    className="stroke-yellow-400 flow-line" strokeWidth="3" />
                  <polygon points="295,25 305,30 295,35" className="fill-yellow-400" />
                  <text x="250" y="50" className="fill-yellow-400 text-xs" textAnchor="middle">
                    θ⁻ ← θ
                  </text>
                </>
              )}
            </svg>

            <div className={`p-4 rounded-xl border-2 transition-all ${
              showLoss ? 'border-rose-500 bg-rose-500/10' : 'border-slate-700 bg-slate-900/50'
            }`}>
              <div className="text-xs text-rose-400 font-bold mb-3 tracking-wider">LOSS COMPUTATION</div>
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className={`p-3 rounded-lg transition-all ${showTarget ? 'bg-cyan-900/50' : 'bg-slate-800/50'}`}>
                  <div className="text-xs text-cyan-400 mb-1">Target (from θ⁻)</div>
                  <div className="font-mono text-lg text-cyan-300">
                    {showTarget ? `y = ${targetValue}` : '—'}
                  </div>
                </div>
                
                <div className={`p-3 rounded-lg transition-all ${showLoss ? 'bg-emerald-900/50' : 'bg-slate-800/50'}`}>
                  <div className="text-xs text-emerald-400 mb-1">Prediction (from θ)</div>
                  <div className="font-mono text-lg text-emerald-300">
                    {showLoss ? `Q = ${qPrediction}` : '—'}
                  </div>
                </div>
                
                <div className={`p-3 rounded-lg transition-all ${showLoss ? 'bg-rose-900/50' : 'bg-slate-800/50'}`}>
                  <div className="text-xs text-rose-400 mb-1">TD Loss</div>
                  <div className="font-mono text-lg text-rose-300">
                    {showLoss ? `L = ${lossValue}` : '—'}
                  </div>
                </div>
              </div>
              
              {showLoss && (
                <div className="mt-3 p-2 bg-slate-800 rounded font-mono text-xs text-center text-slate-400">
                  L = (y - Q(s,a;θ))² = ({targetValue} - {qPrediction})² = {lossValue}
                </div>
              )}
            </div>
          </div>

          <div className="col-span-3">
            <div className="p-4 rounded-xl border-2 border-slate-700 bg-slate-900/50 h-full">
              <div className="text-xs text-slate-400 font-bold mb-3 tracking-wider">KEY INSIGHT</div>
              
              <div className="space-y-4 text-sm">
                <div className={`p-3 rounded-lg transition-all ${
                  currentPhase === 'compute_target' ? 'bg-cyan-900/30 border border-cyan-500/50' : 'bg-slate-800/30'
                }`}>
                  <div className="text-cyan-400 font-semibold mb-1">🧊 Target Network is Frozen</div>
                  <div className="text-slate-400 text-xs">
                    The target y uses old weights θ⁻ that don't change during training. This provides a stable target to learn against.
                  </div>
                </div>
                
                <div className={`p-3 rounded-lg transition-all ${
                  currentPhase === 'update' ? 'bg-emerald-900/30 border border-emerald-500/50' : 'bg-slate-800/30'
                }`}>
                  <div className="text-emerald-400 font-semibold mb-1">🔄 Only Online Updates</div>
                  <div className="text-slate-400 text-xs">
                    Gradient descent only changes θ (online network). The target network θ⁻ stays fixed.
                  </div>
                </div>
                
                <div className={`p-3 rounded-lg transition-all ${
                  currentPhase === 'check_sync' && syncCounter >= SYNC_INTERVAL - 1
                    ? 'bg-yellow-900/30 border border-yellow-500/50' 
                    : 'bg-slate-800/30'
                }`}>
                  <div className="text-yellow-400 font-semibold mb-1">⚡ Periodic Sync</div>
                  <div className="text-slate-400 text-xs">
                    Every {SYNC_INTERVAL} steps, we copy θ → θ⁻. This slowly incorporates learning into the target.
                  </div>
                </div>
                
                <div className={`p-3 rounded-lg transition-all ${
                  currentPhase === 'sample' ? 'bg-amber-900/30 border border-amber-500/50' : 'bg-slate-800/30'
                }`}>
                  <div className="text-amber-400 font-semibold mb-1">🎲 Random Sampling</div>
                  <div className="text-slate-400 text-xs">
                    Replay buffer breaks correlations. Each batch contains diverse experiences from different times.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-xl border border-slate-700 bg-slate-900/30">
          <div className="text-xs text-slate-500 font-bold mb-2">WHY TWO NETWORKS?</div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="p-3 bg-red-900/20 rounded-lg border border-red-500/30">
              <div className="text-red-400 font-semibold mb-1">❌ Without Target Network</div>
              <div className="text-slate-400 text-xs">
                Target y = r + γ·max Q(s';θ) changes every update. We chase a moving target → unstable, diverges.
              </div>
            </div>
            <div className="p-3 bg-green-900/20 rounded-lg border border-green-500/30">
              <div className="text-green-400 font-semibold mb-1">✓ With Target Network</div>
              <div className="text-slate-400 text-xs">
                Target y = r + γ·max Q(s';θ⁻) is fixed for many steps. Like supervised learning with stable labels → converges.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DQNVisualization;
