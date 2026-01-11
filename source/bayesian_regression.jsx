import React, { useState, useCallback, useMemo } from 'react';

// Matrix operations for Bayesian regression
const matMul = (A, B) => {
  const rowsA = A.length, colsA = A[0].length, colsB = B[0].length;
  const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
};

const transpose = (A) => A[0].map((_, i) => A.map(row => row[i]));

const inverse2x2 = (M) => {
  const [[a, b], [c, d]] = M;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-10) return [[1e10, 0], [0, 1e10]];
  return [[d / det, -b / det], [-c / det, a / det]];
};

const addMat = (A, B) => A.map((row, i) => row.map((val, j) => val + B[i][j]));

// Gaussian PDF for 2D
const gaussian2D = (x, y, mean, cov) => {
  const dx = x - mean[0], dy = y - mean[1];
  const invCov = inverse2x2(cov);
  const exponent = -0.5 * (dx * (invCov[0][0] * dx + invCov[0][1] * dy) + 
                          dy * (invCov[1][0] * dx + invCov[1][1] * dy));
  return Math.exp(exponent);
};

export default function BayesianLinearRegression() {
  const [dataPoints, setDataPoints] = useState([
    { x: 0.2, y: 0.3 },
    { x: 0.8, y: 0.75 }
  ]);
  const [priorVar, setPriorVar] = useState(1.0);
  const [noiseVar, setNoiseVar] = useState(0.05);
  const [showPrior, setShowPrior] = useState(true);
  const [showPosterior, setShowPosterior] = useState(true);
  const [showSamples, setShowSamples] = useState(true);
  const [numSamples, setNumSamples] = useState(5);

  // Bayesian linear regression computation
  const bayesianRegression = useMemo(() => {
    const priorMean = [0.5, 0];
    const priorCov = [[priorVar, 0], [0, priorVar]];
    const priorPrec = inverse2x2(priorCov);

    if (dataPoints.length === 0) {
      return { priorMean, priorCov, posteriorMean: priorMean, posteriorCov: priorCov };
    }

    // Design matrix [1, x] for each point
    const X = dataPoints.map(p => [1, p.x]);
    const y = dataPoints.map(p => [p.y]);
    const Xt = transpose(X);
    
    // Posterior precision = prior precision + (1/noise_var) * X^T * X
    const XtX = matMul(Xt, X);
    const scaledXtX = XtX.map(row => row.map(v => v / noiseVar));
    const posteriorPrec = addMat(priorPrec, scaledXtX);
    const posteriorCov = inverse2x2(posteriorPrec);
    
    // Posterior mean = posterior_cov * (prior_prec * prior_mean + (1/noise_var) * X^T * y)
    const priorTerm = matMul(priorPrec, [[priorMean[0]], [priorMean[1]]]);
    const Xty = matMul(Xt, y);
    const dataTerm = Xty.map(row => row.map(v => v / noiseVar));
    const combined = addMat(priorTerm, dataTerm);
    const posteriorMeanVec = matMul(posteriorCov, combined);
    const posteriorMean = [posteriorMeanVec[0][0], posteriorMeanVec[1][0]];

    return { priorMean, priorCov, posteriorMean, posteriorCov };
  }, [dataPoints, priorVar, noiseVar]);

  // Generate samples from posterior
  const posteriorSamples = useMemo(() => {
    const { posteriorMean, posteriorCov } = bayesianRegression;
    const samples = [];
    
    // Cholesky-like decomposition for 2x2
    const a = Math.sqrt(Math.max(0.0001, posteriorCov[0][0]));
    const b = posteriorCov[0][1] / a;
    const c = Math.sqrt(Math.max(0.0001, posteriorCov[1][1] - b * b));
    
    for (let i = 0; i < numSamples; i++) {
      // Box-Muller transform
      const u1 = Math.random(), u2 = Math.random();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      
      const w0 = posteriorMean[0] + a * z1;
      const w1 = posteriorMean[1] + b * z1 + c * z2;
      samples.push({ w0, w1, color: `hsl(${(i * 360 / numSamples)}, 70%, 50%)` });
    }
    return samples;
  }, [bayesianRegression, numSamples, dataPoints]);

  // Predictive distribution
  const getPredictive = useCallback((x) => {
    const { posteriorMean, posteriorCov } = bayesianRegression;
    const phi = [1, x];
    const mean = posteriorMean[0] + posteriorMean[1] * x;
    const var_pred = noiseVar + 
      phi[0] * (posteriorCov[0][0] * phi[0] + posteriorCov[0][1] * phi[1]) +
      phi[1] * (posteriorCov[1][0] * phi[0] + posteriorCov[1][1] * phi[1]);
    return { mean, std: Math.sqrt(Math.max(0.001, var_pred)) };
  }, [bayesianRegression, noiseVar]);

  // Handle click to add point
  const handleDataPlotClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = 1 - (e.clientY - rect.top) / rect.height;
    setDataPoints([...dataPoints, { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) }]);
  };

  const clearPoints = () => setDataPoints([]);
  const removeLastPoint = () => setDataPoints(dataPoints.slice(0, -1));

  // Weight space density grid
  const weightSpaceGrid = useMemo(() => {
    const { priorMean, priorCov, posteriorMean, posteriorCov } = bayesianRegression;
    const gridSize = 40;
    const range = { w0: [-1, 2], w1: [-2, 2] };
    const grid = [];
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const w0 = range.w0[0] + (range.w0[1] - range.w0[0]) * i / (gridSize - 1);
        const w1 = range.w1[0] + (range.w1[1] - range.w1[0]) * j / (gridSize - 1);
        const priorDensity = gaussian2D(w0, w1, priorMean, priorCov);
        const posteriorDensity = gaussian2D(w0, w1, posteriorMean, posteriorCov);
        grid.push({ w0, w1, i, j, priorDensity, posteriorDensity });
      }
    }
    return { grid, gridSize, range };
  }, [bayesianRegression]);

  // Generate prediction curve points
  const predictionCurve = useMemo(() => {
    const points = [];
    for (let i = 0; i <= 50; i++) {
      const x = i / 50;
      const { mean, std } = getPredictive(x);
      points.push({ x, mean, upper1: mean + std, lower1: mean - std, 
                   upper2: mean + 2*std, lower2: mean - 2*std });
    }
    return points;
  }, [getPredictive]);

  const toDataX = (x) => 50 + x * 300;
  const toDataY = (y) => 250 - y * 200;
  const toWeightX = (w0) => 50 + ((w0 - weightSpaceGrid.range.w0[0]) / 
    (weightSpaceGrid.range.w0[1] - weightSpaceGrid.range.w0[0])) * 200;
  const toWeightY = (w1) => 200 - ((w1 - weightSpaceGrid.range.w1[0]) / 
    (weightSpaceGrid.range.w1[1] - weightSpaceGrid.range.w1[0])) * 180;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-center mb-2 text-cyan-400">
          Bayesian Linear Regression
        </h1>
        <p className="text-center text-slate-400 text-sm mb-4">
          Click on the left plot to add observations and watch the uncertainty reduce
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          {/* Data Space Plot */}
          <div className="bg-slate-800 rounded-xl p-4">
            <h2 className="text-lg font-semibold mb-2 text-emerald-400">
              Data Space & Predictive Uncertainty
            </h2>
            <svg 
              viewBox="0 0 400 280" 
              className="w-full cursor-crosshair bg-slate-900 rounded-lg"
              onClick={handleDataPlotClick}
            >
              {/* Grid */}
              {[0, 0.25, 0.5, 0.75, 1].map(v => (
                <g key={v}>
                  <line x1={toDataX(v)} y1={toDataY(0)} x2={toDataX(v)} y2={toDataY(1)} 
                        stroke="#334155" strokeWidth="1" />
                  <line x1={toDataX(0)} y1={toDataY(v)} x2={toDataX(1)} y2={toDataY(v)} 
                        stroke="#334155" strokeWidth="1" />
                  <text x={toDataX(v)} y={toDataY(0) + 15} fill="#64748b" fontSize="10" textAnchor="middle">{v}</text>
                  <text x={toDataX(0) - 10} y={toDataY(v) + 3} fill="#64748b" fontSize="10" textAnchor="end">{v}</text>
                </g>
              ))}
              
              {/* 2-sigma confidence band */}
              <path
                d={`M ${predictionCurve.map(p => `${toDataX(p.x)},${toDataY(p.upper2)}`).join(' L ')} 
                    L ${predictionCurve.slice().reverse().map(p => `${toDataX(p.x)},${toDataY(p.lower2)}`).join(' L ')} Z`}
                fill="rgba(59, 130, 246, 0.15)"
              />
              
              {/* 1-sigma confidence band */}
              <path
                d={`M ${predictionCurve.map(p => `${toDataX(p.x)},${toDataY(p.upper1)}`).join(' L ')} 
                    L ${predictionCurve.slice().reverse().map(p => `${toDataX(p.x)},${toDataY(p.lower1)}`).join(' L ')} Z`}
                fill="rgba(59, 130, 246, 0.25)"
              />

              {/* Sample lines */}
              {showSamples && posteriorSamples.map((s, i) => (
                <line key={i}
                  x1={toDataX(0)} y1={toDataY(s.w0)}
                  x2={toDataX(1)} y2={toDataY(s.w0 + s.w1)}
                  stroke={s.color} strokeWidth="1.5" opacity="0.6"
                />
              ))}

              {/* Mean prediction */}
              <path
                d={`M ${predictionCurve.map(p => `${toDataX(p.x)},${toDataY(p.mean)}`).join(' L ')}`}
                fill="none" stroke="#3b82f6" strokeWidth="2.5"
              />

              {/* Data points */}
              {dataPoints.map((p, i) => (
                <g key={i}>
                  <circle cx={toDataX(p.x)} cy={toDataY(p.y)} r="8" fill="rgba(239, 68, 68, 0.3)" />
                  <circle cx={toDataX(p.x)} cy={toDataY(p.y)} r="5" fill="#ef4444" stroke="#fff" strokeWidth="1.5" />
                </g>
              ))}

              {/* Labels */}
              <text x="200" y="275" fill="#94a3b8" fontSize="11" textAnchor="middle">x</text>
              <text x="25" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" transform="rotate(-90, 25, 140)">y</text>
            </svg>
            
            <div className="flex gap-2 mt-3 flex-wrap">
              <button onClick={clearPoints} 
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors">
                Clear All
              </button>
              <button onClick={removeLastPoint} 
                      className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-sm transition-colors">
                Remove Last
              </button>
              <div className="flex items-center gap-2 ml-auto">
                <label className="text-xs text-slate-400">Samples:</label>
                <input type="checkbox" checked={showSamples} onChange={e => setShowSamples(e.target.checked)} 
                       className="accent-emerald-500" />
              </div>
            </div>
          </div>

          {/* Weight Space Plot */}
          <div className="bg-slate-800 rounded-xl p-4">
            <h2 className="text-lg font-semibold mb-2 text-purple-400">
              Weight Space Distribution
            </h2>
            <svg viewBox="0 0 300 220" className="w-full bg-slate-900 rounded-lg">
              {/* Posterior density */}
              {showPosterior && weightSpaceGrid.grid.map((cell, idx) => {
                const maxDensity = Math.max(...weightSpaceGrid.grid.map(c => c.posteriorDensity));
                const intensity = cell.posteriorDensity / maxDensity;
                if (intensity < 0.01) return null;
                const cellWidth = 200 / weightSpaceGrid.gridSize;
                const cellHeight = 180 / weightSpaceGrid.gridSize;
                return (
                  <rect key={`post-${idx}`}
                    x={50 + cell.i * cellWidth}
                    y={20 + (weightSpaceGrid.gridSize - 1 - cell.j) * cellHeight}
                    width={cellWidth + 0.5} height={cellHeight + 0.5}
                    fill={`rgba(168, 85, 247, ${intensity * 0.8})`}
                  />
                );
              })}

              {/* Prior density (contour) */}
              {showPrior && weightSpaceGrid.grid.map((cell, idx) => {
                const maxDensity = Math.max(...weightSpaceGrid.grid.map(c => c.priorDensity));
                const intensity = cell.priorDensity / maxDensity;
                if (intensity < 0.1 || intensity > 0.15) return null;
                const cellWidth = 200 / weightSpaceGrid.gridSize;
                const cellHeight = 180 / weightSpaceGrid.gridSize;
                return (
                  <rect key={`prior-${idx}`}
                    x={50 + cell.i * cellWidth}
                    y={20 + (weightSpaceGrid.gridSize - 1 - cell.j) * cellHeight}
                    width={cellWidth} height={cellHeight}
                    fill="none" stroke="rgba(34, 197, 94, 0.6)" strokeWidth="0.5"
                  />
                );
              })}

              {/* Axes */}
              <line x1="50" y1="200" x2="250" y2="200" stroke="#64748b" strokeWidth="1" />
              <line x1="50" y1="20" x2="50" y2="200" stroke="#64748b" strokeWidth="1" />
              
              {/* Axis labels */}
              <text x="150" y="215" fill="#94a3b8" fontSize="11" textAnchor="middle">w₀ (intercept)</text>
              <text x="15" y="110" fill="#94a3b8" fontSize="11" textAnchor="middle" transform="rotate(-90, 15, 110)">w₁ (slope)</text>

              {/* Tick marks */}
              {[-1, 0, 1, 2].map(v => (
                <g key={`w0-${v}`}>
                  <line x1={toWeightX(v)} y1="200" x2={toWeightX(v)} y2="205" stroke="#64748b" />
                  <text x={toWeightX(v)} y="212" fill="#64748b" fontSize="8" textAnchor="middle">{v}</text>
                </g>
              ))}
              {[-2, -1, 0, 1, 2].map(v => (
                <g key={`w1-${v}`}>
                  <line x1="45" y1={toWeightY(v)} x2="50" y2={toWeightY(v)} stroke="#64748b" />
                  <text x="42" y={toWeightY(v) + 3} fill="#64748b" fontSize="8" textAnchor="end">{v}</text>
                </g>
              ))}

              {/* Posterior mean marker */}
              <circle 
                cx={toWeightX(bayesianRegression.posteriorMean[0])} 
                cy={toWeightY(bayesianRegression.posteriorMean[1])} 
                r="6" fill="#a855f7" stroke="#fff" strokeWidth="2"
              />

              {/* Sample points */}
              {showSamples && posteriorSamples.map((s, i) => (
                <circle key={i}
                  cx={toWeightX(s.w0)} cy={toWeightY(s.w1)}
                  r="4" fill={s.color} stroke="#fff" strokeWidth="1" opacity="0.8"
                />
              ))}

              {/* Legend */}
              <g transform="translate(255, 30)">
                <rect x="0" y="0" width="10" height="10" fill="rgba(168, 85, 247, 0.6)" />
                <text x="15" y="9" fill="#94a3b8" fontSize="9">Posterior</text>
                <rect x="0" y="15" width="10" height="10" fill="none" stroke="rgba(34, 197, 94, 0.6)" />
                <text x="15" y="24" fill="#94a3b8" fontSize="9">Prior</text>
              </g>
            </svg>

            <div className="flex gap-4 mt-3 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showPrior} onChange={e => setShowPrior(e.target.checked)}
                       className="accent-emerald-500" />
                <span className="text-emerald-400">Prior</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showPosterior} onChange={e => setShowPosterior(e.target.checked)}
                       className="accent-purple-500" />
                <span className="text-purple-400">Posterior</span>
              </label>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-slate-800 rounded-xl p-4">
          <h2 className="text-lg font-semibold mb-3 text-amber-400">Parameters</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Prior Variance (σ²₀): {priorVar.toFixed(2)}
              </label>
              <input type="range" min="0.1" max="3" step="0.1" value={priorVar}
                     onChange={e => setPriorVar(parseFloat(e.target.value))}
                     className="w-full accent-amber-500" />
              <p className="text-xs text-slate-500 mt-1">Controls spread of prior belief about weights</p>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Noise Variance (σ²ₙ): {noiseVar.toFixed(3)}
              </label>
              <input type="range" min="0.001" max="0.2" step="0.001" value={noiseVar}
                     onChange={e => setNoiseVar(parseFloat(e.target.value))}
                     className="w-full accent-amber-500" />
              <p className="text-xs text-slate-500 mt-1">Expected noise in observations</p>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">
                Posterior Samples: {numSamples}
              </label>
              <input type="range" min="1" max="15" step="1" value={numSamples}
                     onChange={e => setNumSamples(parseInt(e.target.value))}
                     className="w-full accent-amber-500" />
              <p className="text-xs text-slate-500 mt-1">Number of sampled regression lines</p>
            </div>
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-slate-800 rounded-xl p-4">
            <h3 className="font-semibold text-emerald-400 mb-2">📊 Predictive Uncertainty</h3>
            <p className="text-sm text-slate-400">
              The shaded bands show ±1σ and ±2σ predictive uncertainty. Notice how bands 
              <span className="text-cyan-400"> narrow near data points</span> and 
              <span className="text-amber-400"> widen away</span> from them.
            </p>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <h3 className="font-semibold text-purple-400 mb-2">🎯 Weight Space</h3>
            <p className="text-sm text-slate-400">
              The posterior (purple) concentrates as you add data. Each point in weight space 
              corresponds to a possible line y = w₀ + w₁x.
            </p>
          </div>
          <div className="bg-slate-800 rounded-xl p-4">
            <h3 className="font-semibold text-amber-400 mb-2">💡 Try This</h3>
            <p className="text-sm text-slate-400">
              Add points at the edges (x≈0 or x≈1) to reduce slope uncertainty. 
              Add clustered points to see how uncertainty remains high elsewhere.
            </p>
          </div>
        </div>

        {/* Current Stats */}
        <div className="bg-slate-800 rounded-xl p-4 mt-4">
          <h3 className="font-semibold text-cyan-400 mb-2">Current Posterior Estimate</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-400">w₀ (intercept):</span>
              <span className="ml-2 text-white font-mono">{bayesianRegression.posteriorMean[0].toFixed(3)}</span>
            </div>
            <div>
              <span className="text-slate-400">w₁ (slope):</span>
              <span className="ml-2 text-white font-mono">{bayesianRegression.posteriorMean[1].toFixed(3)}</span>
            </div>
            <div>
              <span className="text-slate-400">σ(w₀):</span>
              <span className="ml-2 text-white font-mono">{Math.sqrt(bayesianRegression.posteriorCov[0][0]).toFixed(3)}</span>
            </div>
            <div>
              <span className="text-slate-400">σ(w₁):</span>
              <span className="ml-2 text-white font-mono">{Math.sqrt(bayesianRegression.posteriorCov[1][1]).toFixed(3)}</span>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            {dataPoints.length} observation{dataPoints.length !== 1 ? 's' : ''} • 
            Model: y = w₀ + w₁x + ε, where ε ~ N(0, σ²ₙ)
          </p>
        </div>
      </div>
    </div>
  );
}
