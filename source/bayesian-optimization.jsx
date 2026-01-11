import React, { useState, useEffect, useCallback, useMemo } from 'react';

// Gaussian PDF and CDF helpers
const gaussianPDF = (x, mean, std) => {
  const exp = Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
  return exp / (std * Math.sqrt(2 * Math.PI));
};

const gaussianCDF = (x, mean, std) => {
  const z = (x - mean) / std;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989422804 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
};

// True function to optimize (unknown to the optimizer)
const trueFunction = (x) => {
  return Math.sin(x * 2) * Math.cos(x * 0.5) + 0.5 * Math.sin(x * 3);
};

// GP Posterior approximation
const computeGPPosterior = (x, observations, lengthScale = 1.0, noiseVar = 0.01) => {
  if (observations.length === 0) {
    return { mean: 0, std: 1.5 };
  }
  
  const kernel = (x1, x2) => Math.exp(-0.5 * Math.pow((x1 - x2) / lengthScale, 2));
  
  const K = observations.map((o1, i) => 
    observations.map((o2, j) => kernel(o1.x, o2.x) + (i === j ? noiseVar : 0))
  );
  
  const kStar = observations.map(o => kernel(x, o.x));
  const kStarStar = kernel(x, x) + noiseVar;
  
  // Simple matrix inversion for small matrices
  const n = observations.length;
  if (n === 1) {
    const alpha = kStar[0] / K[0][0];
    const mean = alpha * observations[0].y;
    const variance = Math.max(0.001, kStarStar - alpha * kStar[0]);
    return { mean, std: Math.sqrt(variance) };
  }
  
  // For larger matrices, use Cholesky-like approximation
  let mean = 0;
  let totalWeight = 0;
  
  observations.forEach((obs, i) => {
    const dist = Math.abs(x - obs.x);
    const weight = Math.exp(-dist * dist / (2 * lengthScale * lengthScale));
    mean += weight * obs.y;
    totalWeight += weight;
  });
  
  mean = totalWeight > 0 ? mean / totalWeight : 0;
  
  const minDist = Math.min(...observations.map(o => Math.abs(x - o.x)));
  const std = Math.max(0.05, 1.5 * (1 - Math.exp(-minDist / lengthScale)));
  
  return { mean, std };
};

// Acquisition functions
const computeUCB = (mean, std, beta = 2.0) => mean + beta * std;

const computeEI = (mean, std, bestY) => {
  if (std < 0.001) return 0;
  const z = (mean - bestY) / std;
  return (mean - bestY) * gaussianCDF(z, 0, 1) + std * gaussianPDF(z, 0, 1);
};

const computePI = (mean, std, bestY, xi = 0.01) => {
  if (std < 0.001) return 0;
  const z = (mean - bestY - xi) / std;
  return gaussianCDF(z, 0, 1);
};

// Multi-armed bandit helpers
const sampleBeta = (alpha, beta) => {
  // Simple approximation using gamma variates
  const gammaAlpha = alpha + Math.random() * 0.5;
  const gammaBeta = beta + Math.random() * 0.5;
  return gammaAlpha / (gammaAlpha + gammaBeta);
};

export default function BayesianOptimizationViz() {
  const [activeTab, setActiveTab] = useState('acquisition');
  
  // Acquisition function state
  const [acqObservations, setAcqObservations] = useState([
    { x: 1, y: trueFunction(1) },
    { x: 4, y: trueFunction(4) },
  ]);
  const [selectedAcq, setSelectedAcq] = useState('all');
  const [betaParam, setBetaParam] = useState(2.0);
  
  // Sequential optimization state
  const [seqObservations, setSeqObservations] = useState([]);
  const [seqStep, setSeqStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [seqAcqType, setSeqAcqType] = useState('UCB');
  
  // Multi-armed bandit state
  const [arms, setArms] = useState([
    { id: 0, trueProb: 0.3, wins: 0, losses: 0, color: '#ff6b6b' },
    { id: 1, trueProb: 0.5, wins: 0, losses: 0, color: '#4ecdc4' },
    { id: 2, trueProb: 0.7, wins: 0, losses: 0, color: '#ffe66d' },
    { id: 3, trueProb: 0.4, wins: 0, losses: 0, color: '#95e1d3' },
  ]);
  const [banditHistory, setBanditHistory] = useState([]);
  const [banditStrategy, setBanditStrategy] = useState('thompson');
  const [totalPulls, setTotalPulls] = useState(0);
  const [cumulativeRegret, setCumulativeRegret] = useState([]);
  
  // Compute GP posterior and acquisition values
  const xRange = useMemo(() => {
    const points = [];
    for (let x = 0; x <= 6; x += 0.05) {
      points.push(x);
    }
    return points;
  }, []);
  
  const computeAllValues = useCallback((observations) => {
    const bestY = observations.length > 0 ? Math.max(...observations.map(o => o.y)) : -Infinity;
    
    return xRange.map(x => {
      const { mean, std } = computeGPPosterior(x, observations);
      return {
        x,
        true: trueFunction(x),
        mean,
        std,
        upper: mean + 1.96 * std,
        lower: mean - 1.96 * std,
        ucb: computeUCB(mean, std, betaParam),
        ei: computeEI(mean, std, bestY) * 5,
        pi: computePI(mean, std, bestY) * 3,
      };
    });
  }, [xRange, betaParam]);
  
  const acqValues = useMemo(() => computeAllValues(acqObservations), [computeAllValues, acqObservations]);
  const seqValues = useMemo(() => computeAllValues(seqObservations), [computeAllValues, seqObservations]);
  
  // Find next query point
  const findNextQuery = useCallback((values, acqType) => {
    let maxVal = -Infinity;
    let maxX = 3;
    
    values.forEach(v => {
      let val;
      switch(acqType) {
        case 'UCB': val = v.ucb; break;
        case 'EI': val = v.ei; break;
        case 'PI': val = v.pi; break;
        default: val = v.ucb;
      }
      if (val > maxVal) {
        maxVal = val;
        maxX = v.x;
      }
    });
    
    return maxX;
  }, []);
  
  // Sequential optimization step
  const stepOptimization = useCallback(() => {
    const nextX = findNextQuery(seqValues, seqAcqType);
    const nextY = trueFunction(nextX) + (Math.random() - 0.5) * 0.1;
    setSeqObservations(prev => [...prev, { x: nextX, y: nextY }]);
    setSeqStep(prev => prev + 1);
  }, [seqValues, seqAcqType, findNextQuery]);
  
  // Auto-animate sequential optimization
  useEffect(() => {
    if (isAnimating && seqStep < 15) {
      const timer = setTimeout(stepOptimization, 800);
      return () => clearTimeout(timer);
    } else if (seqStep >= 15) {
      setIsAnimating(false);
    }
  }, [isAnimating, seqStep, stepOptimization]);
  
  // Reset sequential optimization
  const resetSequential = () => {
    setSeqObservations([]);
    setSeqStep(0);
    setIsAnimating(false);
  };
  
  // Bandit arm selection
  const selectArm = useCallback((strategy) => {
    let selectedIdx = 0;
    
    switch(strategy) {
      case 'thompson':
        // Thompson Sampling
        const samples = arms.map(arm => sampleBeta(arm.wins + 1, arm.losses + 1));
        selectedIdx = samples.indexOf(Math.max(...samples));
        break;
      case 'ucb':
        // UCB1
        const ucbValues = arms.map((arm, i) => {
          const n = arm.wins + arm.losses;
          if (n === 0) return Infinity;
          const mean = arm.wins / n;
          const bonus = Math.sqrt(2 * Math.log(totalPulls + 1) / n);
          return mean + bonus;
        });
        selectedIdx = ucbValues.indexOf(Math.max(...ucbValues));
        break;
      case 'greedy':
        // Epsilon-greedy
        if (Math.random() < 0.1) {
          selectedIdx = Math.floor(Math.random() * arms.length);
        } else {
          const means = arms.map(arm => {
            const n = arm.wins + arm.losses;
            return n === 0 ? 0.5 : arm.wins / n;
          });
          selectedIdx = means.indexOf(Math.max(...means));
        }
        break;
      default:
        selectedIdx = 0;
    }
    
    return selectedIdx;
  }, [arms, totalPulls]);
  
  const pullArm = useCallback(() => {
    const selectedIdx = selectArm(banditStrategy);
    const arm = arms[selectedIdx];
    const reward = Math.random() < arm.trueProb ? 1 : 0;
    
    setArms(prev => prev.map((a, i) => 
      i === selectedIdx 
        ? { ...a, wins: a.wins + reward, losses: a.losses + (1 - reward) }
        : a
    ));
    
    setBanditHistory(prev => [...prev, { armId: selectedIdx, reward }]);
    setTotalPulls(prev => prev + 1);
    
    // Calculate regret (best arm prob - selected arm prob)
    const bestProb = Math.max(...arms.map(a => a.trueProb));
    const regret = bestProb - arm.trueProb;
    setCumulativeRegret(prev => [...prev, (prev[prev.length - 1] || 0) + regret]);
  }, [selectArm, banditStrategy, arms]);
  
  const resetBandit = () => {
    setArms(prev => prev.map(a => ({ ...a, wins: 0, losses: 0 })));
    setBanditHistory([]);
    setTotalPulls(0);
    setCumulativeRegret([]);
  };
  
  // Auto-pull for bandits
  const [autoPull, setAutoPull] = useState(false);
  
  useEffect(() => {
    if (autoPull && totalPulls < 200) {
      const timer = setTimeout(pullArm, 50);
      return () => clearTimeout(timer);
    } else if (totalPulls >= 200) {
      setAutoPull(false);
    }
  }, [autoPull, totalPulls, pullArm]);
  
  // SVG helpers
  const width = 700;
  const height = 320;
  const margin = { top: 30, right: 30, bottom: 40, left: 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  
  const xScale = (x) => margin.left + (x / 6) * plotWidth;
  const yScale = (y) => margin.top + plotHeight - ((y + 2) / 4) * plotHeight;
  const yScaleAcq = (y) => margin.top + plotHeight - (y / 4) * plotHeight;
  
  const pathFromPoints = (points, xKey, yKey, scaleY = yScale) => {
    return points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p[xKey])} ${scaleY(p[yKey])}`).join(' ');
  };
  
  // Add observation on click
  const handlePlotClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const x = ((clickX - margin.left) / plotWidth) * 6;
    if (x >= 0 && x <= 6) {
      const y = trueFunction(x) + (Math.random() - 0.5) * 0.1;
      setAcqObservations(prev => [...prev, { x, y }]);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%)',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      color: '#e0e0e0',
      padding: '24px',
      boxSizing: 'border-box',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        * { box-sizing: border-box; }
        
        .tab-btn {
          padding: 12px 24px;
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.03);
          color: #888;
          cursor: pointer;
          transition: all 0.3s ease;
          font-family: inherit;
          font-size: 13px;
          letter-spacing: 0.5px;
        }
        
        .tab-btn:hover {
          background: rgba(255,255,255,0.08);
          color: #fff;
        }
        
        .tab-btn.active {
          background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(168, 85, 247, 0.3));
          border-color: rgba(139, 92, 246, 0.5);
          color: #fff;
          box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
        }
        
        .control-btn {
          padding: 10px 20px;
          border: 1px solid rgba(99, 102, 241, 0.4);
          background: rgba(99, 102, 241, 0.1);
          color: #a5b4fc;
          cursor: pointer;
          transition: all 0.2s ease;
          font-family: inherit;
          font-size: 12px;
          border-radius: 6px;
        }
        
        .control-btn:hover {
          background: rgba(99, 102, 241, 0.25);
          border-color: rgba(99, 102, 241, 0.6);
          transform: translateY(-1px);
        }
        
        .control-btn:active {
          transform: translateY(0);
        }
        
        .control-btn.primary {
          background: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(168, 85, 247, 0.4));
          border-color: rgba(139, 92, 246, 0.6);
        }
        
        .control-btn.danger {
          border-color: rgba(239, 68, 68, 0.4);
          background: rgba(239, 68, 68, 0.1);
          color: #fca5a5;
        }
        
        .control-btn.danger:hover {
          background: rgba(239, 68, 68, 0.25);
        }
        
        .panel {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 12px;
          padding: 24px;
          backdrop-filter: blur(10px);
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          background: rgba(255,255,255,0.03);
          border-radius: 6px;
          font-size: 11px;
          letter-spacing: 0.3px;
        }
        
        .slider-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .slider-container input[type="range"] {
          -webkit-appearance: none;
          width: 120px;
          height: 4px;
          background: rgba(255,255,255,0.1);
          border-radius: 2px;
          outline: none;
        }
        
        .slider-container input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          background: #8b5cf6;
          border-radius: 50%;
          cursor: pointer;
        }
        
        .select-styled {
          padding: 8px 12px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.15);
          color: #e0e0e0;
          font-family: inherit;
          font-size: 12px;
          border-radius: 6px;
          cursor: pointer;
          outline: none;
        }
        
        .select-styled:focus {
          border-color: rgba(139, 92, 246, 0.5);
        }
        
        .arm-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 10px;
          padding: 16px;
          transition: all 0.3s ease;
        }
        
        .arm-card:hover {
          background: rgba(255,255,255,0.05);
          transform: translateY(-2px);
        }
        
        .arm-card.selected {
          border-color: rgba(139, 92, 246, 0.6);
          box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        .animating {
          animation: pulse 0.8s ease-in-out infinite;
        }
        
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in {
          animation: slideIn 0.4s ease-out;
        }
      `}</style>
      
      {/* Header */}
      <div style={{ maxWidth: 900, margin: '0 auto 32px' }}>
        <h1 style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: 32,
          fontWeight: 700,
          margin: 0,
          background: 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #10b981 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          letterSpacing: '-0.5px',
        }}>
          Bayesian Optimization
        </h1>
        <p style={{ 
          color: '#666', 
          fontSize: 14, 
          marginTop: 8,
          fontWeight: 300,
        }}>
          Interactive exploration of acquisition functions, sequential optimization & multi-armed bandits
        </p>
      </div>
      
      {/* Tabs */}
      <div style={{ 
        maxWidth: 900, 
        margin: '0 auto 24px',
        display: 'flex',
        gap: 0,
        borderRadius: 8,
        overflow: 'hidden',
      }}>
        <button 
          className={`tab-btn ${activeTab === 'acquisition' ? 'active' : ''}`}
          onClick={() => setActiveTab('acquisition')}
          style={{ borderRadius: '8px 0 0 8px' }}
        >
          ◈ Acquisition Functions
        </button>
        <button 
          className={`tab-btn ${activeTab === 'sequential' ? 'active' : ''}`}
          onClick={() => setActiveTab('sequential')}
        >
          ▷ Sequential Optimization
        </button>
        <button 
          className={`tab-btn ${activeTab === 'bandits' ? 'active' : ''}`}
          onClick={() => setActiveTab('bandits')}
          style={{ borderRadius: '0 8px 8px 0' }}
        >
          ⬡ Multi-Armed Bandits
        </button>
      </div>
      
      {/* Content */}
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        
        {/* Acquisition Functions Tab */}
        {activeTab === 'acquisition' && (
          <div className="panel slide-in">
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'flex-start',
              marginBottom: 20,
              flexWrap: 'wrap',
              gap: 16,
            }}>
              <div>
                <h2 style={{ 
                  fontFamily: "'Space Grotesk', sans-serif",
                  fontSize: 18, 
                  fontWeight: 600, 
                  margin: '0 0 8px 0',
                  color: '#fff',
                }}>
                  Acquisition Function Comparison
                </h2>
                <p style={{ fontSize: 12, color: '#666', margin: 0 }}>
                  Click on the plot to add observations • Compare UCB, EI, and PI strategies
                </p>
              </div>
              
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <select 
                  className="select-styled"
                  value={selectedAcq}
                  onChange={(e) => setSelectedAcq(e.target.value)}
                >
                  <option value="all">Show All</option>
                  <option value="UCB">UCB Only</option>
                  <option value="EI">EI Only</option>
                  <option value="PI">PI Only</option>
                </select>
                
                <div className="slider-container">
                  <span style={{ fontSize: 11, color: '#888' }}>β:</span>
                  <input 
                    type="range" 
                    min="0.5" 
                    max="4" 
                    step="0.1"
                    value={betaParam}
                    onChange={(e) => setBetaParam(parseFloat(e.target.value))}
                  />
                  <span style={{ fontSize: 11, color: '#a5b4fc', minWidth: 30 }}>{betaParam.toFixed(1)}</span>
                </div>
                
                <button 
                  className="control-btn danger"
                  onClick={() => setAcqObservations([{ x: 1, y: trueFunction(1) }, { x: 4, y: trueFunction(4) }])}
                >
                  Reset
                </button>
              </div>
            </div>
            
            {/* GP Plot */}
            <svg 
              width={width} 
              height={height} 
              style={{ display: 'block', margin: '0 auto', cursor: 'crosshair' }}
              onClick={handlePlotClick}
            >
              <defs>
                <linearGradient id="confidenceGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
                  <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.1" />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.3" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
              
              {/* Grid */}
              {[0, 1, 2, 3, 4, 5, 6].map(x => (
                <line 
                  key={`gx${x}`}
                  x1={xScale(x)} y1={margin.top} 
                  x2={xScale(x)} y2={height - margin.bottom}
                  stroke="rgba(255,255,255,0.05)" 
                />
              ))}
              {[-2, -1, 0, 1, 2].map(y => (
                <line 
                  key={`gy${y}`}
                  x1={margin.left} y1={yScale(y)} 
                  x2={width - margin.right} y2={yScale(y)}
                  stroke="rgba(255,255,255,0.05)" 
                />
              ))}
              
              {/* Confidence interval */}
              <path
                d={`${pathFromPoints(acqValues, 'x', 'upper')} ${pathFromPoints([...acqValues].reverse(), 'x', 'lower').replace('M', 'L')} Z`}
                fill="url(#confidenceGrad)"
              />
              
              {/* True function */}
              <path
                d={pathFromPoints(acqValues, 'x', 'true')}
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="6,4"
                opacity="0.6"
              />
              
              {/* GP Mean */}
              <path
                d={pathFromPoints(acqValues, 'x', 'mean')}
                fill="none"
                stroke="#8b5cf6"
                strokeWidth="2.5"
                filter="url(#glow)"
              />
              
              {/* Observations */}
              {acqObservations.map((obs, i) => (
                <g key={i}>
                  <circle
                    cx={xScale(obs.x)}
                    cy={yScale(obs.y)}
                    r="8"
                    fill="rgba(6, 182, 212, 0.2)"
                  />
                  <circle
                    cx={xScale(obs.x)}
                    cy={yScale(obs.y)}
                    r="5"
                    fill="#06b6d4"
                    stroke="#fff"
                    strokeWidth="2"
                  />
                </g>
              ))}
              
              {/* Axis labels */}
              <text x={width / 2} y={height - 8} fill="#666" fontSize="11" textAnchor="middle">x</text>
              <text x={15} y={height / 2} fill="#666" fontSize="11" textAnchor="middle" transform={`rotate(-90, 15, ${height/2})`}>f(x)</text>
            </svg>
            
            {/* Acquisition Functions Plot */}
            <div style={{ marginTop: 24 }}>
              <h3 style={{ 
                fontSize: 14, 
                fontWeight: 500, 
                color: '#888',
                marginBottom: 12,
                fontFamily: "'Space Grotesk', sans-serif",
              }}>
                Acquisition Values
              </h3>
              <svg width={width} height={180} style={{ display: 'block', margin: '0 auto' }}>
                {/* Grid */}
                {[0, 1, 2, 3, 4, 5, 6].map(x => (
                  <line 
                    key={`agx${x}`}
                    x1={xScale(x)} y1={30} 
                    x2={xScale(x)} y2={150}
                    stroke="rgba(255,255,255,0.05)" 
                  />
                ))}
                
                {/* UCB */}
                {(selectedAcq === 'all' || selectedAcq === 'UCB') && (
                  <path
                    d={pathFromPoints(acqValues, 'x', 'ucb', yScaleAcq)}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="2"
                    opacity={selectedAcq === 'all' ? 0.8 : 1}
                  />
                )}
                
                {/* EI */}
                {(selectedAcq === 'all' || selectedAcq === 'EI') && (
                  <path
                    d={pathFromPoints(acqValues, 'x', 'ei', yScaleAcq)}
                    fill="none"
                    stroke="#ec4899"
                    strokeWidth="2"
                    opacity={selectedAcq === 'all' ? 0.8 : 1}
                  />
                )}
                
                {/* PI */}
                {(selectedAcq === 'all' || selectedAcq === 'PI') && (
                  <path
                    d={pathFromPoints(acqValues, 'x', 'pi', yScaleAcq)}
                    fill="none"
                    stroke="#14b8a6"
                    strokeWidth="2"
                    opacity={selectedAcq === 'all' ? 0.8 : 1}
                  />
                )}
                
                {/* Next query markers */}
                {['UCB', 'EI', 'PI'].map(type => {
                  if (selectedAcq !== 'all' && selectedAcq !== type) return null;
                  const nextX = findNextQuery(acqValues, type);
                  const colors = { UCB: '#f59e0b', EI: '#ec4899', PI: '#14b8a6' };
                  return (
                    <g key={type}>
                      <line
                        x1={xScale(nextX)} y1={30}
                        x2={xScale(nextX)} y2={150}
                        stroke={colors[type]}
                        strokeWidth="1"
                        strokeDasharray="4,4"
                        opacity="0.5"
                      />
                      <polygon
                        points={`${xScale(nextX)},155 ${xScale(nextX)-6},165 ${xScale(nextX)+6},165`}
                        fill={colors[type]}
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
            
            {/* Legend */}
            <div style={{ 
              display: 'flex', 
              gap: 16, 
              marginTop: 20,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}>
              <div className="legend-item">
                <div style={{ width: 16, height: 3, background: '#10b981', opacity: 0.6 }} />
                <span>True f(x)</span>
              </div>
              <div className="legend-item">
                <div style={{ width: 16, height: 3, background: '#8b5cf6' }} />
                <span>GP Mean</span>
              </div>
              <div className="legend-item">
                <div style={{ width: 10, height: 10, background: '#06b6d4', borderRadius: '50%' }} />
                <span>Observations</span>
              </div>
              <div className="legend-item">
                <div style={{ width: 16, height: 3, background: '#f59e0b' }} />
                <span>UCB (β={betaParam})</span>
              </div>
              <div className="legend-item">
                <div style={{ width: 16, height: 3, background: '#ec4899' }} />
                <span>Expected Improvement</span>
              </div>
              <div className="legend-item">
                <div style={{ width: 16, height: 3, background: '#14b8a6' }} />
                <span>Probability of Improvement</span>
              </div>
            </div>
            
            {/* Explanation */}
            <div style={{ 
              marginTop: 24, 
              padding: 16, 
              background: 'rgba(139, 92, 246, 0.05)',
              borderRadius: 8,
              border: '1px solid rgba(139, 92, 246, 0.1)',
            }}>
              <p style={{ fontSize: 12, color: '#a0a0a0', margin: 0, lineHeight: 1.7 }}>
                <strong style={{ color: '#f59e0b' }}>UCB</strong> (Upper Confidence Bound) balances exploitation (high mean) with exploration (high uncertainty) via the β parameter.
                <strong style={{ color: '#ec4899', marginLeft: 8 }}>EI</strong> (Expected Improvement) measures expected gain over current best.
                <strong style={{ color: '#14b8a6', marginLeft: 8 }}>PI</strong> (Probability of Improvement) measures likelihood of improving.
                Arrows indicate where each acquisition function suggests querying next.
              </p>
            </div>
          </div>
        )}
        
        {/* Sequential Optimization Tab */}
        {activeTab === 'sequential' && (
          <div className="panel slide-in">
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'flex-start',
              marginBottom: 20,
              flexWrap: 'wrap',
              gap: 16,
            }}>
              <div>
                <h2 style={{ 
                  fontFamily: "'Space Grotesk', sans-serif",
                  fontSize: 18, 
                  fontWeight: 600, 
                  margin: '0 0 8px 0',
                  color: '#fff',
                }}>
                  Sequential Optimization Process
                </h2>
                <p style={{ fontSize: 12, color: '#666', margin: 0 }}>
                  Step through Bayesian optimization iteration by iteration
                </p>
              </div>
              
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <select 
                  className="select-styled"
                  value={seqAcqType}
                  onChange={(e) => setSeqAcqType(e.target.value)}
                >
                  <option value="UCB">UCB Strategy</option>
                  <option value="EI">EI Strategy</option>
                  <option value="PI">PI Strategy</option>
                </select>
                
                <button 
                  className="control-btn primary"
                  onClick={stepOptimization}
                  disabled={isAnimating || seqStep >= 15}
                >
                  Step →
                </button>
                
                <button 
                  className={`control-btn ${isAnimating ? 'animating' : ''}`}
                  onClick={() => setIsAnimating(!isAnimating)}
                  disabled={seqStep >= 15}
                >
                  {isAnimating ? '◼ Pause' : '▶ Auto'}
                </button>
                
                <button 
                  className="control-btn danger"
                  onClick={resetSequential}
                >
                  Reset
                </button>
              </div>
            </div>
            
            {/* Progress indicator */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 8, 
              marginBottom: 16,
            }}>
              <span style={{ fontSize: 11, color: '#888' }}>Iteration:</span>
              <div style={{ 
                flex: 1, 
                height: 4, 
                background: 'rgba(255,255,255,0.1)', 
                borderRadius: 2,
                overflow: 'hidden',
              }}>
                <div style={{ 
                  width: `${(seqStep / 15) * 100}%`, 
                  height: '100%', 
                  background: 'linear-gradient(90deg, #8b5cf6, #06b6d4)',
                  transition: 'width 0.3s ease',
                }} />
              </div>
              <span style={{ fontSize: 13, color: '#a5b4fc', fontWeight: 500 }}>{seqStep}/15</span>
            </div>
            
            {/* Optimization Plot */}
            <svg width={width} height={height} style={{ display: 'block', margin: '0 auto' }}>
              <defs>
                <linearGradient id="seqConfGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                  <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.1" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3" />
                </linearGradient>
              </defs>
              
              {/* Grid */}
              {[0, 1, 2, 3, 4, 5, 6].map(x => (
                <line 
                  key={`sgx${x}`}
                  x1={xScale(x)} y1={margin.top} 
                  x2={xScale(x)} y2={height - margin.bottom}
                  stroke="rgba(255,255,255,0.05)" 
                />
              ))}
              
              {/* Confidence interval */}
              {seqObservations.length > 0 && (
                <path
                  d={`${pathFromPoints(seqValues, 'x', 'upper')} ${pathFromPoints([...seqValues].reverse(), 'x', 'lower').replace('M', 'L')} Z`}
                  fill="url(#seqConfGrad)"
                />
              )}
              
              {/* True function */}
              <path
                d={pathFromPoints(seqValues, 'x', 'true')}
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="6,4"
                opacity="0.6"
              />
              
              {/* GP Mean */}
              {seqObservations.length > 0 && (
                <path
                  d={pathFromPoints(seqValues, 'x', 'mean')}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2.5"
                />
              )}
              
              {/* Observations with order */}
              {seqObservations.map((obs, i) => (
                <g key={i}>
                  <circle
                    cx={xScale(obs.x)}
                    cy={yScale(obs.y)}
                    r="12"
                    fill={`rgba(139, 92, 246, ${0.1 + i * 0.05})`}
                  />
                  <circle
                    cx={xScale(obs.x)}
                    cy={yScale(obs.y)}
                    r="6"
                    fill={i === seqObservations.length - 1 ? '#f59e0b' : '#8b5cf6'}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={xScale(obs.x)}
                    y={yScale(obs.y) - 16}
                    fill="#888"
                    fontSize="10"
                    textAnchor="middle"
                  >
                    {i + 1}
                  </text>
                </g>
              ))}
              
              {/* Next query indicator */}
              {seqObservations.length > 0 && seqStep < 15 && (
                <g>
                  <line
                    x1={xScale(findNextQuery(seqValues, seqAcqType))}
                    y1={margin.top}
                    x2={xScale(findNextQuery(seqValues, seqAcqType))}
                    y2={height - margin.bottom}
                    stroke="#f59e0b"
                    strokeWidth="2"
                    strokeDasharray="8,4"
                    opacity="0.6"
                  />
                  <text
                    x={xScale(findNextQuery(seqValues, seqAcqType))}
                    y={margin.top - 8}
                    fill="#f59e0b"
                    fontSize="11"
                    textAnchor="middle"
                  >
                    Next Query
                  </text>
                </g>
              )}
              
              {/* Best found marker */}
              {seqObservations.length > 0 && (
                <g>
                  {(() => {
                    const best = seqObservations.reduce((a, b) => a.y > b.y ? a : b);
                    return (
                      <>
                        <circle
                          cx={xScale(best.x)}
                          cy={yScale(best.y)}
                          r="16"
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="2"
                          strokeDasharray="4,2"
                        />
                        <text
                          x={xScale(best.x) + 20}
                          y={yScale(best.y) + 4}
                          fill="#10b981"
                          fontSize="11"
                        >
                          Best
                        </text>
                      </>
                    );
                  })()}
                </g>
              )}
              
              {/* Start message */}
              {seqObservations.length === 0 && (
                <text
                  x={width / 2}
                  y={height / 2}
                  fill="#666"
                  fontSize="14"
                  textAnchor="middle"
                >
                  Click "Step" or "Auto" to begin optimization
                </text>
              )}
            </svg>
            
            {/* Stats */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: 16,
              marginTop: 24,
            }}>
              <div style={{ 
                padding: 16, 
                background: 'rgba(16, 185, 129, 0.05)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#10b981', marginBottom: 4 }}>Best Value Found</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {seqObservations.length > 0 
                    ? Math.max(...seqObservations.map(o => o.y)).toFixed(3) 
                    : '—'}
                </div>
              </div>
              <div style={{ 
                padding: 16, 
                background: 'rgba(139, 92, 246, 0.05)',
                border: '1px solid rgba(139, 92, 246, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#8b5cf6', marginBottom: 4 }}>Best Location</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {seqObservations.length > 0 
                    ? `x = ${seqObservations.reduce((a, b) => a.y > b.y ? a : b).x.toFixed(2)}` 
                    : '—'}
                </div>
              </div>
              <div style={{ 
                padding: 16, 
                background: 'rgba(245, 158, 11, 0.05)',
                border: '1px solid rgba(245, 158, 11, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#f59e0b', marginBottom: 4 }}>Global Maximum</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  ≈ 1.34 at x ≈ 4.7
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Multi-Armed Bandits Tab */}
        {activeTab === 'bandits' && (
          <div className="panel slide-in">
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'flex-start',
              marginBottom: 20,
              flexWrap: 'wrap',
              gap: 16,
            }}>
              <div>
                <h2 style={{ 
                  fontFamily: "'Space Grotesk', sans-serif",
                  fontSize: 18, 
                  fontWeight: 600, 
                  margin: '0 0 8px 0',
                  color: '#fff',
                }}>
                  Multi-Armed Bandit Problem
                </h2>
                <p style={{ fontSize: 12, color: '#666', margin: 0 }}>
                  Watch exploration vs exploitation strategies in action
                </p>
              </div>
              
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <select 
                  className="select-styled"
                  value={banditStrategy}
                  onChange={(e) => setBanditStrategy(e.target.value)}
                >
                  <option value="thompson">Thompson Sampling</option>
                  <option value="ucb">UCB1</option>
                  <option value="greedy">ε-Greedy (ε=0.1)</option>
                </select>
                
                <button 
                  className="control-btn primary"
                  onClick={pullArm}
                  disabled={autoPull}
                >
                  Pull Arm
                </button>
                
                <button 
                  className={`control-btn ${autoPull ? 'animating' : ''}`}
                  onClick={() => setAutoPull(!autoPull)}
                  disabled={totalPulls >= 200}
                >
                  {autoPull ? '◼ Stop' : '▶ Auto (×200)'}
                </button>
                
                <button 
                  className="control-btn danger"
                  onClick={resetBandit}
                >
                  Reset
                </button>
              </div>
            </div>
            
            {/* Slot Machine Arms */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 16,
              marginBottom: 24,
            }}>
              {arms.map((arm, i) => {
                const n = arm.wins + arm.losses;
                const estProb = n > 0 ? arm.wins / n : 0.5;
                const isLastSelected = banditHistory.length > 0 && banditHistory[banditHistory.length - 1].armId === i;
                
                return (
                  <div 
                    key={arm.id}
                    className={`arm-card ${isLastSelected ? 'selected' : ''}`}
                    style={{ 
                      borderColor: isLastSelected ? arm.color : undefined,
                      boxShadow: isLastSelected ? `0 0 20px ${arm.color}40` : undefined,
                    }}
                  >
                    <div style={{ 
                      fontSize: 28, 
                      textAlign: 'center', 
                      marginBottom: 12,
                      filter: isLastSelected ? 'drop-shadow(0 0 8px currentColor)' : 'none',
                      color: arm.color,
                    }}>
                      🎰
                    </div>
                    <div style={{ 
                      fontSize: 14, 
                      fontWeight: 600, 
                      textAlign: 'center',
                      color: arm.color,
                      marginBottom: 8,
                    }}>
                      Arm {i + 1}
                    </div>
                    
                    {/* Win rate bar */}
                    <div style={{ 
                      height: 6, 
                      background: 'rgba(255,255,255,0.1)', 
                      borderRadius: 3,
                      overflow: 'hidden',
                      marginBottom: 8,
                    }}>
                      <div style={{ 
                        width: `${estProb * 100}%`, 
                        height: '100%', 
                        background: arm.color,
                        transition: 'width 0.2s ease',
                      }} />
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                      <span style={{ color: '#888' }}>Est: {(estProb * 100).toFixed(1)}%</span>
                      <span style={{ color: '#666' }}>{n} pulls</span>
                    </div>
                    
                    <div style={{ 
                      fontSize: 10, 
                      color: '#555', 
                      textAlign: 'center',
                      marginTop: 8,
                      fontStyle: 'italic',
                    }}>
                      True: {(arm.trueProb * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Regret Chart */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ 
                fontSize: 14, 
                fontWeight: 500, 
                color: '#888',
                marginBottom: 12,
                fontFamily: "'Space Grotesk', sans-serif",
              }}>
                Cumulative Regret Over Time
              </h3>
              <svg width={width} height={180} style={{ display: 'block', margin: '0 auto' }}>
                {/* Background */}
                <rect x={margin.left} y={20} width={plotWidth} height={130} fill="rgba(255,255,255,0.02)" rx="4" />
                
                {/* Grid lines */}
                {[0, 50, 100, 150, 200].map(x => (
                  <line
                    key={`rx${x}`}
                    x1={margin.left + (x / 200) * plotWidth}
                    y1={20}
                    x2={margin.left + (x / 200) * plotWidth}
                    y2={150}
                    stroke="rgba(255,255,255,0.05)"
                  />
                ))}
                
                {/* Regret line */}
                {cumulativeRegret.length > 1 && (
                  <path
                    d={cumulativeRegret.map((r, i) => {
                      const x = margin.left + (i / 200) * plotWidth;
                      const maxRegret = Math.max(...cumulativeRegret, 1);
                      const y = 150 - (r / Math.max(maxRegret, 20)) * 120;
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="2"
                  />
                )}
                
                {/* Linear regret reference */}
                {cumulativeRegret.length > 1 && (
                  <line
                    x1={margin.left}
                    y1={150}
                    x2={margin.left + plotWidth}
                    y2={150 - 120}
                    stroke="rgba(255,255,255,0.1)"
                    strokeDasharray="4,4"
                  />
                )}
                
                {/* Labels */}
                <text x={margin.left} y={165} fill="#666" fontSize="10">0</text>
                <text x={margin.left + plotWidth} y={165} fill="#666" fontSize="10" textAnchor="end">200</text>
                <text x={width / 2} y={175} fill="#666" fontSize="10" textAnchor="middle">Pulls</text>
                
                {cumulativeRegret.length === 0 && (
                  <text x={width / 2} y={90} fill="#555" fontSize="12" textAnchor="middle">
                    Start pulling arms to see regret accumulation
                  </text>
                )}
              </svg>
            </div>
            
            {/* Stats */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: 16,
            }}>
              <div style={{ 
                padding: 16, 
                background: 'rgba(239, 68, 68, 0.05)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#ef4444', marginBottom: 4 }}>Cumulative Regret</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {cumulativeRegret.length > 0 
                    ? cumulativeRegret[cumulativeRegret.length - 1].toFixed(2) 
                    : '0.00'}
                </div>
              </div>
              <div style={{ 
                padding: 16, 
                background: 'rgba(78, 205, 196, 0.05)',
                border: '1px solid rgba(78, 205, 196, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#4ecdc4', marginBottom: 4 }}>Total Pulls</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {totalPulls}
                </div>
              </div>
              <div style={{ 
                padding: 16, 
                background: 'rgba(255, 230, 109, 0.05)',
                border: '1px solid rgba(255, 230, 109, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#ffe66d', marginBottom: 4 }}>Best Arm Pulls</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {arms[2].wins + arms[2].losses} ({totalPulls > 0 ? ((arms[2].wins + arms[2].losses) / totalPulls * 100).toFixed(0) : 0}%)
                </div>
              </div>
              <div style={{ 
                padding: 16, 
                background: 'rgba(16, 185, 129, 0.05)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, color: '#10b981', marginBottom: 4 }}>Win Rate</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                  {totalPulls > 0 
                    ? (banditHistory.filter(h => h.reward === 1).length / totalPulls * 100).toFixed(1) 
                    : 0}%
                </div>
              </div>
            </div>
            
            {/* Strategy explanation */}
            <div style={{ 
              marginTop: 24, 
              padding: 16, 
              background: 'rgba(139, 92, 246, 0.05)',
              borderRadius: 8,
              border: '1px solid rgba(139, 92, 246, 0.1)',
            }}>
              <p style={{ fontSize: 12, color: '#a0a0a0', margin: 0, lineHeight: 1.7 }}>
                {banditStrategy === 'thompson' && (
                  <>
                    <strong style={{ color: '#8b5cf6' }}>Thompson Sampling</strong> maintains a Beta distribution belief over each arm's win probability.
                    It samples from each distribution and plays the arm with highest sample—naturally balancing exploration (uncertain arms) with exploitation (high-probability arms).
                  </>
                )}
                {banditStrategy === 'ucb' && (
                  <>
                    <strong style={{ color: '#8b5cf6' }}>UCB1</strong> selects the arm with highest upper confidence bound: estimated mean + exploration bonus.
                    The bonus shrinks with more pulls, ensuring under-explored arms get tried while focusing on promising ones.
                  </>
                )}
                {banditStrategy === 'greedy' && (
                  <>
                    <strong style={{ color: '#8b5cf6' }}>ε-Greedy</strong> exploits the best-known arm 90% of the time and explores randomly 10% of the time.
                    Simple but effective—though the fixed exploration rate can lead to linear regret growth.
                  </>
                )}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
