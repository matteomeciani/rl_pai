import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend, ReferenceLine } from 'recharts';
import { Play, Pause, RotateCcw, ChevronRight, Info } from 'lucide-react';

// Gaussian PDF
const gaussian = (x, mean, std) => {
  const coefficient = 1 / (std * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mean) / std, 2);
  return coefficient * Math.exp(exponent);
};

// True posterior (mixture of Gaussians for interesting shape)
const truePosterior = (x) => {
  return 0.6 * gaussian(x, 2, 0.8) + 0.4 * gaussian(x, -1, 1.2);
};

// KL divergence approximation (numerical integration)
const computeKL = (qMean, qStd) => {
  let kl = 0;
  const dx = 0.1;
  for (let x = -8; x <= 8; x += dx) {
    const q = gaussian(x, qMean, qStd);
    const p = truePosterior(x);
    if (q > 1e-10 && p > 1e-10) {
      kl += q * Math.log(q / p) * dx;
    }
  }
  return Math.max(0, kl);
};

// ELBO computation (simplified)
const computeELBO = (qMean, qStd) => {
  const kl = computeKL(qMean, qStd);
  const logEvidence = 2.5; // Constant for visualization
  return logEvidence - kl;
};

// Optimization step (gradient descent on variational parameters)
const optimizationStep = (qMean, qStd, learningRate = 0.08) => {
  const eps = 0.01;
  
  // Numerical gradients
  const elbo = computeELBO(qMean, qStd);
  const gradMean = (computeELBO(qMean + eps, qStd) - elbo) / eps;
  const gradStd = (computeELBO(qMean, qStd + eps) - elbo) / eps;
  
  // Update with momentum-like behavior
  const newMean = qMean + learningRate * gradMean;
  const newStd = Math.max(0.3, qStd + learningRate * gradStd * 0.5);
  
  return { mean: newMean, std: newStd };
};

export default function VariationalInferenceViz() {
  // Variational parameters
  const [qMean, setQMean] = useState(-3);
  const [qStd, setQStd] = useState(2.5);
  const [iteration, setIteration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [history, setHistory] = useState([]);
  const [speed, setSpeed] = useState(300);
  const [showInfo, setShowInfo] = useState(false);
  
  const maxIterations = 60;
  const intervalRef = useRef(null);

  // Generate distribution data
  const generateDistributionData = useCallback(() => {
    const data = [];
    for (let x = -6; x <= 6; x += 0.1) {
      data.push({
        x: x.toFixed(2),
        q: gaussian(x, qMean, qStd),
        p: truePosterior(x),
      });
    }
    return data;
  }, [qMean, qStd]);

  // Initialize history
  useEffect(() => {
    const initialKL = computeKL(-3, 2.5);
    const initialELBO = computeELBO(-3, 2.5);
    setHistory([{ iteration: 0, kl: initialKL, elbo: initialELBO, mean: -3, std: 2.5 }]);
  }, []);

  // Step function
  const step = useCallback(() => {
    if (iteration >= maxIterations) {
      setIsPlaying(false);
      return;
    }

    const { mean: newMean, std: newStd } = optimizationStep(qMean, qStd);
    const newKL = computeKL(newMean, newStd);
    const newELBO = computeELBO(newMean, newStd);

    setQMean(newMean);
    setQStd(newStd);
    setIteration(prev => prev + 1);
    setHistory(prev => [...prev, {
      iteration: prev.length,
      kl: newKL,
      elbo: newELBO,
      mean: newMean,
      std: newStd
    }]);
  }, [qMean, qStd, iteration]);

  // Auto-play
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(step, speed);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, step, speed]);

  // Reset function
  const reset = () => {
    setIsPlaying(false);
    setQMean(-3);
    setQStd(2.5);
    setIteration(0);
    const initialKL = computeKL(-3, 2.5);
    const initialELBO = computeELBO(-3, 2.5);
    setHistory([{ iteration: 0, kl: initialKL, elbo: initialELBO, mean: -3, std: 2.5 }]);
  };

  const distributionData = generateDistributionData();
  const currentKL = computeKL(qMean, qStd);
  const currentELBO = computeELBO(qMean, qStd);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#e0e0e0',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: '24px',
      boxSizing: 'border-box'
    }}>
      {/* Header */}
      <div style={{ 
        textAlign: 'center', 
        marginBottom: '32px',
        position: 'relative'
      }}>
        <h1 style={{
          fontSize: '2.2rem',
          fontWeight: '300',
          letterSpacing: '0.15em',
          margin: 0,
          background: 'linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textTransform: 'uppercase'
        }}>
          Variational Inference
        </h1>
        <p style={{ 
          color: '#6b7280', 
          marginTop: '8px',
          fontSize: '0.85rem',
          letterSpacing: '0.1em'
        }}>
          Interactive ELBO Optimization & KL Divergence Minimization
        </p>
        <button
          onClick={() => setShowInfo(!showInfo)}
          style={{
            position: 'absolute',
            right: 0,
            top: '50%',
            transform: 'translateY(-50%)',
            background: 'rgba(124, 58, 237, 0.2)',
            border: '1px solid rgba(124, 58, 237, 0.4)',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#a78bfa',
            transition: 'all 0.2s'
          }}
        >
          <Info size={18} />
        </button>
      </div>

      {/* Info Panel */}
      {showInfo && (
        <div style={{
          background: 'rgba(30, 30, 50, 0.95)',
          border: '1px solid rgba(124, 58, 237, 0.3)',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '24px',
          fontSize: '0.85rem',
          lineHeight: '1.7'
        }}>
          <h3 style={{ color: '#a78bfa', marginTop: 0 }}>Understanding Variational Inference</h3>
          <p><strong style={{ color: '#00d4ff' }}>Goal:</strong> Approximate an intractable posterior p(z|x) with a simpler distribution q(z).</p>
          <p><strong style={{ color: '#f472b6' }}>ELBO</strong> (Evidence Lower Bound): We maximize ELBO = E<sub>q</sub>[log p(x,z)] - E<sub>q</sub>[log q(z)], which is equivalent to minimizing KL(q||p).</p>
          <p><strong style={{ color: '#4ade80' }}>KL Divergence:</strong> Measures how different q(z) is from p(z|x). As KL → 0, our approximation improves.</p>
          <p style={{ color: '#6b7280', marginBottom: 0 }}>Watch as q(z) (cyan) converges toward p(z|x) (pink) through gradient-based optimization.</p>
        </div>
      )}

      {/* Main Distribution Visualization */}
      <div style={{
        background: 'rgba(20, 20, 35, 0.8)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '24px',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '16px'
        }}>
          <h2 style={{ 
            margin: 0, 
            fontSize: '1rem', 
            color: '#9ca3af',
            letterSpacing: '0.1em',
            textTransform: 'uppercase'
          }}>
            Distribution Comparison
          </h2>
          <div style={{ display: 'flex', gap: '24px', fontSize: '0.8rem' }}>
            <span style={{ color: '#00d4ff' }}>● q(z) Variational</span>
            <span style={{ color: '#f472b6' }}>● p(z|x) True Posterior</span>
          </div>
        </div>
        
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={distributionData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="qGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="pGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f472b6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#f472b6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              dataKey="x" 
              stroke="#4b5563" 
              tick={{ fill: '#6b7280', fontSize: 11 }}
              tickLine={{ stroke: '#4b5563' }}
              label={{ value: 'z', position: 'right', fill: '#6b7280' }}
            />
            <YAxis 
              stroke="#4b5563" 
              tick={{ fill: '#6b7280', fontSize: 11 }}
              tickLine={{ stroke: '#4b5563' }}
              label={{ value: 'density', angle: -90, position: 'insideLeft', fill: '#6b7280' }}
            />
            <Area 
              type="monotone" 
              dataKey="p" 
              stroke="#f472b6" 
              strokeWidth={2}
              fill="url(#pGradient)" 
              isAnimationActive={false}
            />
            <Area 
              type="monotone" 
              dataKey="q" 
              stroke="#00d4ff" 
              strokeWidth={2.5}
              fill="url(#qGradient)" 
              isAnimationActive={false}
            />
            <ReferenceLine x={qMean.toFixed(2)} stroke="#00d4ff" strokeDasharray="5 5" strokeOpacity={0.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '20px',
        marginBottom: '24px'
      }}>
        {/* ELBO Chart */}
        <div style={{
          background: 'rgba(20, 20, 35, 0.8)',
          borderRadius: '12px',
          padding: '20px',
          border: '1px solid rgba(255,255,255,0.08)'
        }}>
          <h3 style={{ 
            margin: '0 0 12px 0', 
            fontSize: '0.85rem', 
            color: '#4ade80',
            letterSpacing: '0.08em',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span>ELBO MAXIMIZATION ↑</span>
            <span style={{ 
              fontFamily: 'monospace',
              fontSize: '1.1rem',
              color: '#4ade80'
            }}>
              {currentELBO.toFixed(3)}
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="iteration" stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 10 }} domain={['auto', 'auto']} />
              <Line 
                type="monotone" 
                dataKey="elbo" 
                stroke="#4ade80" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* KL Divergence Chart */}
        <div style={{
          background: 'rgba(20, 20, 35, 0.8)',
          borderRadius: '12px',
          padding: '20px',
          border: '1px solid rgba(255,255,255,0.08)'
        }}>
          <h3 style={{ 
            margin: '0 0 12px 0', 
            fontSize: '0.85rem', 
            color: '#f472b6',
            letterSpacing: '0.08em',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span>KL DIVERGENCE ↓</span>
            <span style={{ 
              fontFamily: 'monospace',
              fontSize: '1.1rem',
              color: '#f472b6'
            }}>
              {currentKL.toFixed(3)}
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="iteration" stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 10 }} domain={[0, 'auto']} />
              <Line 
                type="monotone" 
                dataKey="kl" 
                stroke="#f472b6" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Parameter Display */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '16px',
        marginBottom: '24px'
      }}>
        {[
          { label: 'μ (mean)', value: qMean.toFixed(3), color: '#00d4ff' },
          { label: 'σ (std)', value: qStd.toFixed(3), color: '#00d4ff' },
          { label: 'Iteration', value: iteration, color: '#a78bfa' },
          { label: 'Convergence', value: `${Math.max(0, (1 - currentKL / 3) * 100).toFixed(1)}%`, color: '#4ade80' }
        ].map((param, i) => (
          <div key={i} style={{
            background: 'rgba(20, 20, 35, 0.6)',
            borderRadius: '10px',
            padding: '16px',
            textAlign: 'center',
            border: '1px solid rgba(255,255,255,0.06)'
          }}>
            <div style={{ 
              fontSize: '0.7rem', 
              color: '#6b7280', 
              marginBottom: '4px',
              textTransform: 'uppercase',
              letterSpacing: '0.1em'
            }}>
              {param.label}
            </div>
            <div style={{ 
              fontSize: '1.4rem', 
              fontWeight: '600',
              color: param.color,
              fontFamily: "'JetBrains Mono', monospace"
            }}>
              {param.value}
            </div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '16px',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={reset}
          style={{
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: '10px',
            padding: '12px 20px',
            color: '#ef4444',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.85rem',
            fontFamily: 'inherit',
            transition: 'all 0.2s',
            letterSpacing: '0.05em'
          }}
        >
          <RotateCcw size={16} />
          Reset
        </button>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={iteration >= maxIterations}
          style={{
            background: isPlaying 
              ? 'rgba(251, 191, 36, 0.2)' 
              : 'linear-gradient(135deg, rgba(124, 58, 237, 0.3), rgba(0, 212, 255, 0.3))',
            border: `1px solid ${isPlaying ? 'rgba(251, 191, 36, 0.5)' : 'rgba(124, 58, 237, 0.5)'}`,
            borderRadius: '10px',
            padding: '14px 32px',
            color: isPlaying ? '#fbbf24' : '#fff',
            cursor: iteration >= maxIterations ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.9rem',
            fontFamily: 'inherit',
            fontWeight: '500',
            transition: 'all 0.2s',
            letterSpacing: '0.08em',
            opacity: iteration >= maxIterations ? 0.5 : 1
          }}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button
          onClick={step}
          disabled={iteration >= maxIterations || isPlaying}
          style={{
            background: 'rgba(0, 212, 255, 0.15)',
            border: '1px solid rgba(0, 212, 255, 0.4)',
            borderRadius: '10px',
            padding: '12px 20px',
            color: '#00d4ff',
            cursor: (iteration >= maxIterations || isPlaying) ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '0.85rem',
            fontFamily: 'inherit',
            transition: 'all 0.2s',
            letterSpacing: '0.05em',
            opacity: (iteration >= maxIterations || isPlaying) ? 0.5 : 1
          }}
        >
          <ChevronRight size={16} />
          Step
        </button>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(20, 20, 35, 0.6)',
          padding: '8px 16px',
          borderRadius: '10px',
          border: '1px solid rgba(255,255,255,0.06)'
        }}>
          <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>Speed:</span>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={550 - speed}
            onChange={(e) => setSpeed(550 - parseInt(e.target.value))}
            style={{
              width: '80px',
              accentColor: '#7c3aed'
            }}
          />
        </div>
      </div>

      {/* Footer Note */}
      <div style={{
        textAlign: 'center',
        marginTop: '32px',
        fontSize: '0.75rem',
        color: '#4b5563',
        letterSpacing: '0.05em'
      }}>
        q(z) is a Gaussian approximation optimizing toward a bimodal true posterior p(z|x)
      </div>
    </div>
  );
}
