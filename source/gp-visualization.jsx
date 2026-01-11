import React, { useState, useCallback, useMemo, useEffect } from 'react';

// Matrix utilities
const choleskyDecomposition = (A) => {
  const n = A.length;
  const L = Array(n).fill(null).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      if (i === j) {
        L[i][j] = Math.sqrt(Math.max(A[i][i] - sum, 1e-10));
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
};

const solveTriangularLower = (L, b) => {
  const n = L.length;
  const x = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) {
      sum -= L[i][j] * x[j];
    }
    x[i] = sum / L[i][i];
  }
  return x;
};

const solveTriangularUpper = (U, b) => {
  const n = U.length;
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) {
      sum -= U[j][i] * x[j];
    }
    x[i] = sum / U[i][i];
  }
  return x;
};

const matVecMult = (M, v) => M.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));

// Kernel functions
const rbfKernel = (x1, x2, lengthScale, variance) => {
  const diff = x1 - x2;
  return variance * Math.exp(-0.5 * (diff * diff) / (lengthScale * lengthScale));
};

const maternKernel = (x1, x2, lengthScale, variance, nu = 1.5) => {
  const r = Math.abs(x1 - x2) / lengthScale;
  if (r < 1e-10) return variance;
  if (nu === 0.5) {
    return variance * Math.exp(-r);
  } else if (nu === 1.5) {
    const sqrt3r = Math.sqrt(3) * r;
    return variance * (1 + sqrt3r) * Math.exp(-sqrt3r);
  } else {
    const sqrt5r = Math.sqrt(5) * r;
    return variance * (1 + sqrt5r + (5 * r * r) / 3) * Math.exp(-sqrt5r);
  }
};

const periodicKernel = (x1, x2, lengthScale, variance, period = 2) => {
  const diff = Math.abs(x1 - x2);
  const sinTerm = Math.sin(Math.PI * diff / period);
  return variance * Math.exp(-2 * sinTerm * sinTerm / (lengthScale * lengthScale));
};

// Random number generation
const boxMuller = () => {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

const sampleMVN = (mean, L) => {
  const n = mean.length;
  const z = Array(n).fill(0).map(() => boxMuller());
  const Lz = matVecMult(L, z);
  return mean.map((m, i) => m + Lz[i]);
};

export default function GaussianProcessExplorer() {
  const [kernelType, setKernelType] = useState('rbf');
  const [lengthScale, setLengthScale] = useState(0.3);
  const [variance, setVariance] = useState(1.0);
  const [noiseVariance, setNoiseVariance] = useState(0.05);
  const [dataPoints, setDataPoints] = useState([]);
  const [showPriorSamples, setShowPriorSamples] = useState(true);
  const [showPosterior, setShowPosterior] = useState(true);
  const [showEpistemic, setShowEpistemic] = useState(true);
  const [showAleatoric, setShowAleatoric] = useState(true);
  const [priorSeed, setPriorSeed] = useState(0);
  const [activeTab, setActiveTab] = useState('explore');

  const xMin = -3;
  const xMax = 3;
  const nGrid = 100;
  const xGrid = useMemo(() => 
    Array(nGrid).fill(0).map((_, i) => xMin + (xMax - xMin) * i / (nGrid - 1)),
    []
  );

  const getKernel = useCallback((x1, x2) => {
    switch (kernelType) {
      case 'matern12':
        return maternKernel(x1, x2, lengthScale, variance, 0.5);
      case 'matern32':
        return maternKernel(x1, x2, lengthScale, variance, 1.5);
      case 'matern52':
        return maternKernel(x1, x2, lengthScale, variance, 2.5);
      case 'periodic':
        return periodicKernel(x1, x2, lengthScale, variance, 2);
      default:
        return rbfKernel(x1, x2, lengthScale, variance);
    }
  }, [kernelType, lengthScale, variance]);

  // Compute prior covariance
  const priorCov = useMemo(() => {
    const K = xGrid.map(x1 => xGrid.map(x2 => getKernel(x1, x2)));
    // Add small jitter for numerical stability
    for (let i = 0; i < nGrid; i++) {
      K[i][i] += 1e-6;
    }
    return K;
  }, [xGrid, getKernel]);

  // Sample from prior
  const priorSamples = useMemo(() => {
    // Reset random seed for reproducibility
    let seed = priorSeed + kernelType.length + Math.floor(lengthScale * 100) + Math.floor(variance * 100);
    const seededRandom = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    
    const boxMullerSeeded = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    try {
      const L = choleskyDecomposition(priorCov);
      const samples = [];
      for (let s = 0; s < 5; s++) {
        const z = xGrid.map(() => boxMullerSeeded());
        const Lz = matVecMult(L, z);
        samples.push(Lz);
      }
      return samples;
    } catch (e) {
      return [];
    }
  }, [priorCov, xGrid, priorSeed, kernelType, lengthScale, variance]);

  // Compute posterior
  const posterior = useMemo(() => {
    if (dataPoints.length === 0) {
      return {
        mean: xGrid.map(() => 0),
        epistemicVar: xGrid.map((_, i) => priorCov[i][i]),
        aleatoricVar: xGrid.map(() => noiseVariance)
      };
    }

    const X = dataPoints.map(p => p.x);
    const y = dataPoints.map(p => p.y);
    const n = X.length;

    // K(X, X) + noise
    const Kxx = X.map((x1, i) => X.map((x2, j) => 
      getKernel(x1, x2) + (i === j ? noiseVariance : 0)
    ));

    // K(X*, X)
    const Ksx = xGrid.map(xs => X.map(x => getKernel(xs, x)));

    // K(X*, X*)
    const Kss = xGrid.map((xs1, i) => getKernel(xs1, xs1));

    try {
      const L = choleskyDecomposition(Kxx);
      
      // Solve L * alpha_temp = y, then L^T * alpha = alpha_temp
      const alphaTemp = solveTriangularLower(L, y);
      const alpha = solveTriangularUpper(L, alphaTemp);

      // Posterior mean = Ksx * alpha
      const mean = Ksx.map(row => row.reduce((sum, k, i) => sum + k * alpha[i], 0));

      // Posterior variance
      const epistemicVar = xGrid.map((_, i) => {
        const v = solveTriangularLower(L, Ksx[i]);
        const reduction = v.reduce((sum, vi) => sum + vi * vi, 0);
        return Math.max(0, Kss[i] - reduction);
      });

      return {
        mean,
        epistemicVar,
        aleatoricVar: xGrid.map(() => noiseVariance)
      };
    } catch (e) {
      return {
        mean: xGrid.map(() => 0),
        epistemicVar: xGrid.map((_, i) => priorCov[i][i]),
        aleatoricVar: xGrid.map(() => noiseVariance)
      };
    }
  }, [dataPoints, xGrid, getKernel, noiseVariance, priorCov]);

  // SVG dimensions
  const width = 700;
  const height = 400;
  const padding = { top: 30, right: 30, bottom: 50, left: 50 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const xScale = (x) => padding.left + (x - xMin) / (xMax - xMin) * plotWidth;
  const yScale = (y) => padding.top + plotHeight / 2 - y * (plotHeight / 6);
  const xInverse = (px) => xMin + (px - padding.left) / plotWidth * (xMax - xMin);
  const yInverse = (py) => (padding.top + plotHeight / 2 - py) / (plotHeight / 6);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    
    if (px >= padding.left && px <= width - padding.right &&
        py >= padding.top && py <= height - padding.bottom) {
      const x = xInverse(px);
      const y = yInverse(py);
      setDataPoints(prev => [...prev, { x, y, id: Date.now() }]);
    }
  };

  const removePoint = (id) => {
    setDataPoints(prev => prev.filter(p => p.id !== id));
  };

  const pathFromPoints = (ys) => {
    return xGrid.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(ys[i])}`).join(' ');
  };

  const areaPath = (upperYs, lowerYs) => {
    const upper = xGrid.map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(upperYs[i])}`).join(' ');
    const lower = [...xGrid].reverse().map((x, i) => `L ${xScale(x)} ${yScale(lowerYs[xGrid.length - 1 - i])}`).join(' ');
    return `${upper} ${lower} Z`;
  };

  const priorColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6'];

  const kernelDescriptions = {
    rbf: 'Smooth, infinitely differentiable functions. Good default choice.',
    matern12: 'Rough, continuous but not differentiable. Like Brownian motion.',
    matern32: 'Once differentiable. Balances smoothness and flexibility.',
    matern52: 'Twice differentiable. Very commonly used in practice.',
    periodic: 'Repeating patterns. Great for seasonal data.'
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-2 bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
          Gaussian Process Explorer
        </h1>
        <p className="text-slate-400 text-center mb-6">
          Click to add data points • Explore how uncertainty evolves
        </p>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 justify-center">
          {['explore', 'learn'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === tab 
                  ? 'bg-violet-600 text-white' 
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              {tab === 'explore' ? '🔬 Explore' : '📚 Learn'}
            </button>
          ))}
        </div>

        {activeTab === 'learn' ? (
          <div className="bg-slate-900 rounded-xl p-6 space-y-6">
            <section>
              <h2 className="text-xl font-semibold text-violet-400 mb-3">What is a Gaussian Process?</h2>
              <p className="text-slate-300 leading-relaxed">
                A Gaussian Process (GP) is a probability distribution over functions. Instead of learning 
                specific parameters, a GP defines a distribution where any finite collection of function 
                values follows a multivariate Gaussian distribution. This lets us express uncertainty 
                about functions in a principled way.
              </p>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-cyan-400 mb-3">The Kernel: DNA of a GP</h2>
              <p className="text-slate-300 leading-relaxed mb-3">
                The kernel (covariance function) encodes our assumptions about the function we're modeling:
              </p>
              <ul className="space-y-2 text-slate-300">
                <li><span className="text-violet-400 font-medium">Length scale:</span> How quickly the function varies. Small = wiggly, Large = smooth</li>
                <li><span className="text-violet-400 font-medium">Variance:</span> The overall amplitude of function values</li>
                <li><span className="text-violet-400 font-medium">Kernel type:</span> The "texture" of functions (smooth, rough, periodic)</li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-emerald-400 mb-3">Two Types of Uncertainty</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-slate-800 p-4 rounded-lg border border-violet-500/30">
                  <h3 className="font-semibold text-violet-400 mb-2">Epistemic Uncertainty</h3>
                  <p className="text-slate-400 text-sm">
                    Uncertainty about the true function due to limited data. 
                    <span className="text-violet-300 font-medium"> Reducible with more observations!</span>
                    Watch it shrink near data points.
                  </p>
                </div>
                <div className="bg-slate-800 p-4 rounded-lg border border-amber-500/30">
                  <h3 className="font-semibold text-amber-400 mb-2">Aleatoric Uncertainty</h3>
                  <p className="text-slate-400 text-sm">
                    Inherent noise in observations. 
                    <span className="text-amber-300 font-medium"> Cannot be reduced!</span>
                    Even with infinite data, measurements have noise.
                  </p>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-rose-400 mb-3">Try These Experiments</h2>
              <ol className="space-y-2 text-slate-300 list-decimal list-inside">
                <li>Add points and watch epistemic uncertainty collapse locally</li>
                <li>Change length scale: see how assumptions affect predictions far from data</li>
                <li>Increase noise variance: posterior won't pass exactly through points</li>
                <li>Try different kernels: Matérn 1/2 is rough, RBF is infinitely smooth</li>
                <li>Add points, then remove them to see uncertainty return</li>
              </ol>
            </section>
          </div>
        ) : (
          <>
            {/* Main visualization */}
            <div className="bg-slate-900 rounded-xl p-4 mb-6">
              <svg 
                width={width} 
                height={height} 
                className="mx-auto cursor-crosshair"
                onClick={handleClick}
              >
                {/* Grid */}
                <defs>
                  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#334155" strokeWidth="0.5"/>
                  </pattern>
                </defs>
                <rect x={padding.left} y={padding.top} width={plotWidth} height={plotHeight} fill="url(#grid)" />

                {/* Axes */}
                <line x1={padding.left} y1={yScale(0)} x2={width - padding.right} y2={yScale(0)} stroke="#64748b" strokeWidth="1"/>
                <line x1={xScale(0)} y1={padding.top} x2={xScale(0)} y2={height - padding.bottom} stroke="#64748b" strokeWidth="1"/>
                
                {/* Axis labels */}
                <text x={width / 2} y={height - 10} textAnchor="middle" fill="#94a3b8" fontSize="12">x</text>
                <text x={15} y={height / 2} textAnchor="middle" fill="#94a3b8" fontSize="12" transform={`rotate(-90, 15, ${height / 2})`}>f(x)</text>

                {/* Tick marks */}
                {[-2, -1, 0, 1, 2].map(x => (
                  <g key={x}>
                    <line x1={xScale(x)} y1={yScale(0) - 4} x2={xScale(x)} y2={yScale(0) + 4} stroke="#64748b"/>
                    <text x={xScale(x)} y={yScale(0) + 18} textAnchor="middle" fill="#64748b" fontSize="10">{x}</text>
                  </g>
                ))}
                {[-2, -1, 1, 2].map(y => (
                  <g key={y}>
                    <line x1={xScale(0) - 4} y1={yScale(y)} x2={xScale(0) + 4} y2={yScale(y)} stroke="#64748b"/>
                    <text x={xScale(0) - 12} y={yScale(y) + 4} textAnchor="middle" fill="#64748b" fontSize="10">{y}</text>
                  </g>
                ))}

                {/* Prior samples */}
                {showPriorSamples && dataPoints.length === 0 && priorSamples.map((sample, i) => (
                  <path
                    key={i}
                    d={pathFromPoints(sample)}
                    fill="none"
                    stroke={priorColors[i]}
                    strokeWidth="1.5"
                    opacity="0.6"
                  />
                ))}

                {/* Posterior uncertainty bands */}
                {dataPoints.length > 0 && showPosterior && (
                  <>
                    {/* Aleatoric (outer band) */}
                    {showAleatoric && (
                      <path
                        d={areaPath(
                          posterior.mean.map((m, i) => m + 2 * Math.sqrt(posterior.epistemicVar[i] + posterior.aleatoricVar[i])),
                          posterior.mean.map((m, i) => m - 2 * Math.sqrt(posterior.epistemicVar[i] + posterior.aleatoricVar[i]))
                        )}
                        fill="#f59e0b"
                        opacity="0.15"
                      />
                    )}
                    {/* Epistemic (inner band) */}
                    {showEpistemic && (
                      <path
                        d={areaPath(
                          posterior.mean.map((m, i) => m + 2 * Math.sqrt(posterior.epistemicVar[i])),
                          posterior.mean.map((m, i) => m - 2 * Math.sqrt(posterior.epistemicVar[i]))
                        )}
                        fill="#8b5cf6"
                        opacity="0.3"
                      />
                    )}
                    {/* Posterior mean */}
                    <path
                      d={pathFromPoints(posterior.mean)}
                      fill="none"
                      stroke="#8b5cf6"
                      strokeWidth="2.5"
                    />
                  </>
                )}

                {/* Prior mean (zero) when no data */}
                {dataPoints.length === 0 && (
                  <line 
                    x1={padding.left} 
                    y1={yScale(0)} 
                    x2={width - padding.right} 
                    y2={yScale(0)} 
                    stroke="#8b5cf6" 
                    strokeWidth="2" 
                    strokeDasharray="8,4"
                    opacity="0.5"
                  />
                )}

                {/* Data points */}
                {dataPoints.map(point => (
                  <g key={point.id} onClick={(e) => { e.stopPropagation(); removePoint(point.id); }} className="cursor-pointer">
                    <circle
                      cx={xScale(point.x)}
                      cy={yScale(point.y)}
                      r="8"
                      fill="#0f172a"
                      stroke="#22d3ee"
                      strokeWidth="2.5"
                    />
                    <circle
                      cx={xScale(point.x)}
                      cy={yScale(point.y)}
                      r="3"
                      fill="#22d3ee"
                    />
                  </g>
                ))}

                {/* Instructions overlay */}
                {dataPoints.length === 0 && (
                  <text x={width / 2} y={padding.top + 30} textAnchor="middle" fill="#64748b" fontSize="14">
                    Click anywhere to add observations
                  </text>
                )}
              </svg>

              {/* Legend */}
              <div className="flex flex-wrap gap-4 justify-center mt-4 text-sm">
                {dataPoints.length === 0 && showPriorSamples && (
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-0.5 bg-gradient-to-r from-red-500 via-yellow-500 to-blue-500"></div>
                    <span className="text-slate-400">Prior samples</span>
                  </div>
                )}
                {dataPoints.length > 0 && (
                  <>
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-0.5 bg-violet-500"></div>
                      <span className="text-slate-400">Posterior mean</span>
                    </div>
                    {showEpistemic && (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-violet-500/30 rounded"></div>
                        <span className="text-slate-400">Epistemic (±2σ)</span>
                      </div>
                    )}
                    {showAleatoric && (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-amber-500/20 rounded"></div>
                        <span className="text-slate-400">+ Aleatoric</span>
                      </div>
                    )}
                  </>
                )}
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full border-2 border-cyan-400 bg-slate-900"></div>
                  <span className="text-slate-400">Data (click to remove)</span>
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Kernel settings */}
              <div className="bg-slate-900 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-violet-400 mb-4">🧬 Kernel Settings</h3>
                
                <div className="mb-4">
                  <label className="block text-sm text-slate-400 mb-2">Kernel Type</label>
                  <select
                    value={kernelType}
                    onChange={(e) => setKernelType(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-violet-500"
                  >
                    <option value="rbf">RBF (Squared Exponential)</option>
                    <option value="matern12">Matérn 1/2 (Exponential)</option>
                    <option value="matern32">Matérn 3/2</option>
                    <option value="matern52">Matérn 5/2</option>
                    <option value="periodic">Periodic</option>
                  </select>
                  <p className="text-xs text-slate-500 mt-1">{kernelDescriptions[kernelType]}</p>
                </div>

                <div className="mb-4">
                  <label className="block text-sm text-slate-400 mb-2">
                    Length Scale: <span className="text-violet-400 font-mono">{lengthScale.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.05"
                    value={lengthScale}
                    onChange={(e) => setLengthScale(parseFloat(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Wiggly</span>
                    <span>Smooth</span>
                  </div>
                </div>

                <div className="mb-4">
                  <label className="block text-sm text-slate-400 mb-2">
                    Signal Variance: <span className="text-violet-400 font-mono">{variance.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={variance}
                    onChange={(e) => setVariance(parseFloat(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Low amplitude</span>
                    <span>High amplitude</span>
                  </div>
                </div>

                <button
                  onClick={() => setPriorSeed(s => s + 1)}
                  className="w-full py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition-colors"
                >
                  🎲 Resample Prior
                </button>
              </div>

              {/* Uncertainty & display */}
              <div className="bg-slate-900 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-amber-400 mb-4">📊 Uncertainty & Display</h3>

                <div className="mb-4">
                  <label className="block text-sm text-slate-400 mb-2">
                    Noise Variance (σ²ₙ): <span className="text-amber-400 font-mono">{noiseVariance.toFixed(3)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.001"
                    max="0.3"
                    step="0.005"
                    value={noiseVariance}
                    onChange={(e) => setNoiseVariance(parseFloat(e.target.value))}
                    className="w-full accent-amber-500"
                  />
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Near-exact obs</span>
                    <span>Very noisy</span>
                  </div>
                </div>

                <div className="space-y-3 mb-4">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showPriorSamples}
                      onChange={(e) => setShowPriorSamples(e.target.checked)}
                      className="w-4 h-4 accent-violet-500"
                    />
                    <span className="text-slate-300">Show prior samples</span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showPosterior}
                      onChange={(e) => setShowPosterior(e.target.checked)}
                      className="w-4 h-4 accent-violet-500"
                    />
                    <span className="text-slate-300">Show posterior</span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showEpistemic}
                      onChange={(e) => setShowEpistemic(e.target.checked)}
                      className="w-4 h-4 accent-violet-500"
                    />
                    <span className="text-slate-300">Epistemic uncertainty <span className="text-violet-400">(reducible)</span></span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showAleatoric}
                      onChange={(e) => setShowAleatoric(e.target.checked)}
                      className="w-4 h-4 accent-amber-500"
                    />
                    <span className="text-slate-300">Aleatoric uncertainty <span className="text-amber-400">(irreducible)</span></span>
                  </label>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setDataPoints([])}
                    className="flex-1 py-2 bg-rose-900/50 hover:bg-rose-800/50 rounded-lg text-rose-300 transition-colors"
                  >
                    Clear All Points
                  </button>
                  <button
                    onClick={() => {
                      const newPoints = [];
                      for (let i = 0; i < 5; i++) {
                        const x = xMin + Math.random() * (xMax - xMin);
                        const y = Math.sin(2 * x) + (Math.random() - 0.5) * 0.5;
                        newPoints.push({ x, y, id: Date.now() + i });
                      }
                      setDataPoints(newPoints);
                    }}
                    className="flex-1 py-2 bg-cyan-900/50 hover:bg-cyan-800/50 rounded-lg text-cyan-300 transition-colors"
                  >
                    Add Random Data
                  </button>
                </div>
              </div>
            </div>

            {/* Stats panel */}
            {dataPoints.length > 0 && (
              <div className="mt-6 bg-slate-900 rounded-xl p-4">
                <div className="flex flex-wrap gap-6 justify-center text-sm">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-cyan-400">{dataPoints.length}</div>
                    <div className="text-slate-500">Observations</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-violet-400">
                      {(posterior.epistemicVar.reduce((a, b) => a + b, 0) / nGrid).toFixed(3)}
                    </div>
                    <div className="text-slate-500">Avg Epistemic Var</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-amber-400">
                      {noiseVariance.toFixed(3)}
                    </div>
                    <div className="text-slate-500">Aleatoric Var (fixed)</div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
