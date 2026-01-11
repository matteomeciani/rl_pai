import React, { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 5;
const ACTIONS = [
  { name: 'up', dx: 0, dy: -1, arrow: '↑' },
  { name: 'right', dx: 1, dy: 0, arrow: '→' },
  { name: 'down', dx: 0, dy: 1, arrow: '↓' },
  { name: 'left', dx: -1, dy: 0, arrow: '←' },
];

// Initial grid setup
const createInitialGrid = () => {
  const grid = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    const row = [];
    for (let x = 0; x < GRID_SIZE; x++) {
      row.push({
        value: 0,
        reward: -0.04, // Small negative reward to encourage reaching goal
        isWall: false,
        isGoal: false,
        isPit: false,
        policy: null,
      });
    }
    grid.push(row);
  }
  // Set goal state (high reward)
  grid[0][GRID_SIZE - 1].reward = 1;
  grid[0][GRID_SIZE - 1].isGoal = true;
  
  // Set pit (negative reward)
  grid[1][GRID_SIZE - 1].reward = -1;
  grid[1][GRID_SIZE - 1].isPit = true;
  
  // Set walls
  grid[1][1].isWall = true;
  grid[2][1].isWall = true;
  grid[3][3].isWall = true;
  
  return grid;
};

const ValueIterationViz = () => {
  const [grid, setGrid] = useState(createInitialGrid());
  const [gamma, setGamma] = useState(0.9);
  const [iteration, setIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [selectedCell, setSelectedCell] = useState(null);
  const [maxDelta, setMaxDelta] = useState(0);
  const [converged, setConverged] = useState(false);

  // Get next state given action (handles boundaries and walls)
  const getNextState = useCallback((x, y, action, currentGrid) => {
    const newX = x + action.dx;
    const newY = y + action.dy;
    
    // Check boundaries
    if (newX < 0 || newX >= GRID_SIZE || newY < 0 || newY >= GRID_SIZE) {
      return { x, y }; // Stay in place
    }
    
    // Check walls
    if (currentGrid[newY][newX].isWall) {
      return { x, y }; // Stay in place
    }
    
    return { x: newX, y: newY };
  }, []);

  // Perform one iteration of value iteration
  const performIteration = useCallback(() => {
    if (converged) return;
    
    setGrid(prevGrid => {
      const newGrid = prevGrid.map(row => row.map(cell => ({ ...cell })));
      let maxChange = 0;
      
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          const cell = prevGrid[y][x];
          
          // Skip walls, goals, and pits (terminal states)
          if (cell.isWall || cell.isGoal || cell.isPit) {
            newGrid[y][x].value = cell.isGoal ? 1 : (cell.isPit ? -1 : 0);
            continue;
          }
          
          // Calculate Q-value for each action
          let bestValue = -Infinity;
          let bestAction = null;
          
          for (const action of ACTIONS) {
            const nextState = getNextState(x, y, action, prevGrid);
            const nextValue = prevGrid[nextState.y][nextState.x].value;
            const qValue = cell.reward + gamma * nextValue;
            
            if (qValue > bestValue) {
              bestValue = qValue;
              bestAction = action;
            }
          }
          
          const change = Math.abs(bestValue - cell.value);
          maxChange = Math.max(maxChange, change);
          
          newGrid[y][x].value = bestValue;
          newGrid[y][x].policy = bestAction;
        }
      }
      
      setMaxDelta(maxChange);
      if (maxChange < 0.0001) {
        setConverged(true);
        setIsRunning(false);
      }
      
      return newGrid;
    });
    
    setIteration(prev => prev + 1);
  }, [gamma, getNextState, converged]);

  // Auto-run effect
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(() => {
      performIteration();
    }, speed);
    
    return () => clearInterval(interval);
  }, [isRunning, speed, performIteration]);

  // Reset function
  const handleReset = () => {
    setGrid(createInitialGrid());
    setIteration(0);
    setIsRunning(false);
    setSelectedCell(null);
    setMaxDelta(0);
    setConverged(false);
  };

  // Get color based on value
  const getValueColor = (value, cell) => {
    if (cell.isWall) return '#1a1a2e';
    if (cell.isGoal) return '#10b981';
    if (cell.isPit) return '#ef4444';
    
    // Normalize value to color
    const normalized = (value + 1) / 2; // Map -1 to 1 → 0 to 1
    const hue = normalized * 120; // Red (0) to Green (120)
    const saturation = Math.min(Math.abs(value) * 100, 70);
    return `hsl(${hue}, ${saturation}%, ${35 + normalized * 20}%)`;
  };

  return (
    <div className="min-h-screen bg-slate-950" style={{
      padding: '32px',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      color: '#e2e8f0',
    }}>
      {/* Header */}
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '700',
          background: 'linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '8px',
          letterSpacing: '-0.02em',
        }}>
          Value Iteration Visualizer
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '1rem', marginBottom: '32px' }}>
          Watch an agent learn optimal navigation through dynamic programming
        </p>

        <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
          {/* Grid Section */}
          <div style={{ flex: '1', minWidth: '400px' }}>
            <div style={{
              background: 'rgba(30, 41, 59, 0.5)',
              borderRadius: '16px',
              padding: '24px',
              border: '1px solid rgba(148, 163, 184, 0.1)',
              backdropFilter: 'blur(10px)',
            }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`,
                gap: '4px',
                marginBottom: '20px',
              }}>
                {grid.map((row, y) =>
                  row.map((cell, x) => (
                    <div
                      key={`${x}-${y}`}
                      onClick={() => setSelectedCell({ x, y, cell })}
                      style={{
                        aspectRatio: '1',
                        background: getValueColor(cell.value, cell),
                        borderRadius: '8px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        border: selectedCell?.x === x && selectedCell?.y === y
                          ? '3px solid #f472b6'
                          : '1px solid rgba(255,255,255,0.1)',
                        position: 'relative',
                        boxShadow: cell.isGoal 
                          ? '0 0 20px rgba(16, 185, 129, 0.5)' 
                          : cell.isPit 
                            ? '0 0 20px rgba(239, 68, 68, 0.5)' 
                            : 'none',
                      }}
                    >
                      {cell.isWall ? (
                        <span style={{ fontSize: '1.5rem' }}>🧱</span>
                      ) : cell.isGoal ? (
                        <span style={{ fontSize: '1.5rem' }}>🏆</span>
                      ) : cell.isPit ? (
                        <span style={{ fontSize: '1.5rem' }}>💀</span>
                      ) : (
                        <>
                          <span style={{
                            fontSize: '0.85rem',
                            fontWeight: '600',
                            color: '#fff',
                            textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                          }}>
                            {cell.value.toFixed(2)}
                          </span>
                          {cell.policy && (
                            <span style={{
                              fontSize: '1.2rem',
                              marginTop: '2px',
                              color: '#fbbf24',
                              textShadow: '0 0 10px rgba(251, 191, 36, 0.5)',
                            }}>
                              {cell.policy.arrow}
                            </span>
                          )}
                        </>
                      )}
                      {/* Coordinates */}
                      <span style={{
                        position: 'absolute',
                        bottom: '2px',
                        right: '4px',
                        fontSize: '0.6rem',
                        color: 'rgba(255,255,255,0.3)',
                      }}>
                        {x},{y}
                      </span>
                    </div>
                  ))
                )}
              </div>

              {/* Legend */}
              <div style={{
                display: 'flex',
                gap: '16px',
                justifyContent: 'center',
                flexWrap: 'wrap',
                fontSize: '0.8rem',
                color: '#94a3b8',
              }}>
                <span>🏆 Goal (+1)</span>
                <span>💀 Pit (-1)</span>
                <span>🧱 Wall</span>
                <span style={{ color: '#fbbf24' }}>↑→↓← Policy</span>
              </div>
            </div>
          </div>

          {/* Controls & Info Section */}
          <div style={{ width: '320px' }}>
            {/* Stats Panel */}
            <div style={{
              background: 'rgba(30, 41, 59, 0.5)',
              borderRadius: '16px',
              padding: '20px',
              marginBottom: '16px',
              border: '1px solid rgba(148, 163, 184, 0.1)',
            }}>
              <h3 style={{ 
                fontSize: '0.9rem', 
                color: '#60a5fa', 
                marginBottom: '16px',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}>
                Statistics
              </h3>
              <div style={{ display: 'grid', gap: '12px' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '12px',
                  background: 'rgba(96, 165, 250, 0.1)',
                  borderRadius: '8px',
                }}>
                  <span>Iteration</span>
                  <span style={{ 
                    fontWeight: '700', 
                    color: '#60a5fa',
                    fontSize: '1.2rem',
                  }}>{iteration}</span>
                </div>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '12px',
                  background: 'rgba(167, 139, 250, 0.1)',
                  borderRadius: '8px',
                }}>
                  <span>Max Δ</span>
                  <span style={{ 
                    fontWeight: '700', 
                    color: '#a78bfa',
                  }}>{maxDelta.toFixed(6)}</span>
                </div>
                {converged && (
                  <div style={{
                    padding: '12px',
                    background: 'rgba(16, 185, 129, 0.2)',
                    borderRadius: '8px',
                    textAlign: 'center',
                    color: '#10b981',
                    fontWeight: '600',
                  }}>
                    ✓ Converged!
                  </div>
                )}
              </div>
            </div>

            {/* Controls Panel */}
            <div style={{
              background: 'rgba(30, 41, 59, 0.5)',
              borderRadius: '16px',
              padding: '20px',
              marginBottom: '16px',
              border: '1px solid rgba(148, 163, 184, 0.1)',
            }}>
              <h3 style={{ 
                fontSize: '0.9rem', 
                color: '#a78bfa', 
                marginBottom: '16px',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}>
                Controls
              </h3>
              
              {/* Gamma Slider */}
              <div style={{ marginBottom: '20px' }}>
                <label style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  marginBottom: '8px',
                  fontSize: '0.85rem',
                }}>
                  <span>Discount (γ)</span>
                  <span style={{ color: '#f472b6', fontWeight: '600' }}>{gamma.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.99"
                  step="0.01"
                  value={gamma}
                  onChange={(e) => {
                    setGamma(parseFloat(e.target.value));
                    handleReset();
                  }}
                  style={{
                    width: '100%',
                    accentColor: '#f472b6',
                  }}
                />
              </div>

              {/* Speed Slider */}
              <div style={{ marginBottom: '20px' }}>
                <label style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  marginBottom: '8px',
                  fontSize: '0.85rem',
                }}>
                  <span>Speed</span>
                  <span style={{ color: '#fbbf24', fontWeight: '600' }}>{speed}ms</span>
                </label>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={speed}
                  onChange={(e) => setSpeed(parseInt(e.target.value))}
                  style={{
                    width: '100%',
                    accentColor: '#fbbf24',
                  }}
                />
              </div>

              {/* Buttons */}
              <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  disabled={converged}
                  style={{
                    flex: 1,
                    padding: '12px',
                    borderRadius: '8px',
                    border: 'none',
                    background: isRunning 
                      ? 'linear-gradient(135deg, #ef4444, #dc2626)' 
                      : 'linear-gradient(135deg, #10b981, #059669)',
                    color: 'white',
                    fontWeight: '600',
                    cursor: converged ? 'not-allowed' : 'pointer',
                    opacity: converged ? 0.5 : 1,
                    transition: 'all 0.2s ease',
                  }}
                >
                  {isRunning ? '⏸ Pause' : '▶ Run'}
                </button>
                <button
                  onClick={performIteration}
                  disabled={isRunning || converged}
                  style={{
                    flex: 1,
                    padding: '12px',
                    borderRadius: '8px',
                    border: 'none',
                    background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                    color: 'white',
                    fontWeight: '600',
                    cursor: (isRunning || converged) ? 'not-allowed' : 'pointer',
                    opacity: (isRunning || converged) ? 0.5 : 1,
                  }}
                >
                  Step →
                </button>
              </div>
              <button
                onClick={handleReset}
                style={{
                  width: '100%',
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  background: 'transparent',
                  color: '#94a3b8',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                ↺ Reset
              </button>
            </div>

            {/* Selected Cell Info */}
            {selectedCell && (
              <div style={{
                background: 'rgba(30, 41, 59, 0.5)',
                borderRadius: '16px',
                padding: '20px',
                border: '1px solid rgba(244, 114, 182, 0.3)',
              }}>
                <h3 style={{ 
                  fontSize: '0.9rem', 
                  color: '#f472b6', 
                  marginBottom: '12px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                }}>
                  Cell ({selectedCell.x}, {selectedCell.y})
                </h3>
                <div style={{ fontSize: '0.85rem', lineHeight: '1.8' }}>
                  <div><span style={{ color: '#94a3b8' }}>Value:</span> <span style={{ color: '#60a5fa' }}>{selectedCell.cell.value.toFixed(4)}</span></div>
                  <div><span style={{ color: '#94a3b8' }}>Reward:</span> <span style={{ color: '#10b981' }}>{selectedCell.cell.reward}</span></div>
                  <div><span style={{ color: '#94a3b8' }}>Policy:</span> <span style={{ color: '#fbbf24' }}>{selectedCell.cell.policy?.name || 'N/A'}</span></div>
                  <div><span style={{ color: '#94a3b8' }}>Type:</span> {
                    selectedCell.cell.isGoal ? '🏆 Goal' :
                    selectedCell.cell.isPit ? '💀 Pit' :
                    selectedCell.cell.isWall ? '🧱 Wall' : 'Normal'
                  }</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Explanation Section */}
        <div style={{
          marginTop: '32px',
          background: 'rgba(30, 41, 59, 0.5)',
          borderRadius: '16px',
          padding: '24px',
          border: '1px solid rgba(148, 163, 184, 0.1)',
        }}>
          <h3 style={{ 
            fontSize: '1.1rem', 
            color: '#60a5fa', 
            marginBottom: '16px',
          }}>
            How Value Iteration Works
          </h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '20px',
            fontSize: '0.9rem',
            color: '#cbd5e1',
            lineHeight: '1.7',
          }}>
            <div style={{
              padding: '16px',
              background: 'rgba(96, 165, 250, 0.1)',
              borderRadius: '12px',
              borderLeft: '3px solid #60a5fa',
            }}>
              <strong style={{ color: '#60a5fa' }}>1. Bellman Update</strong>
              <p style={{ marginTop: '8px', marginBottom: '0' }}>
                For each state, calculate the maximum expected value over all possible actions:
                <code style={{ 
                  display: 'block', 
                  marginTop: '8px',
                  padding: '8px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                }}>
                  V(s) = max[R(s) + γ·V(s')]
                </code>
              </p>
            </div>
            <div style={{
              padding: '16px',
              background: 'rgba(167, 139, 250, 0.1)',
              borderRadius: '12px',
              borderLeft: '3px solid #a78bfa',
            }}>
              <strong style={{ color: '#a78bfa' }}>2. Policy Extraction</strong>
              <p style={{ marginTop: '8px', marginBottom: '0' }}>
                The optimal policy is the action that achieves the maximum value. Arrows show which direction the agent should move from each cell.
              </p>
            </div>
            <div style={{
              padding: '16px',
              background: 'rgba(244, 114, 182, 0.1)',
              borderRadius: '12px',
              borderLeft: '3px solid #f472b6',
            }}>
              <strong style={{ color: '#f472b6' }}>3. Discount Factor (γ)</strong>
              <p style={{ marginTop: '8px', marginBottom: '0' }}>
                Controls how much future rewards matter. High γ = plan ahead. Low γ = focus on immediate rewards. Try adjusting it!
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValueIterationViz;
