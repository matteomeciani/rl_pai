import React, { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 4;
const ACTIONS = ['↑', '→', '↓', '←'];
const ACTION_DELTAS = { '↑': [-1, 0], '→': [0, 1], '↓': [1, 0], '←': [0, -1] };
const RMAX = 100;
const KNOWN_THRESHOLD = 1;
const GAMMA = 0.9;

const RmaxVisualization = () => {
  const [agentPos, setAgentPos] = useState([0, 0]);
  const [visitCounts, setVisitCounts] = useState({});
  const [transitionCounts, setTransitionCounts] = useState({});
  const [rewardEstimates, setRewardEstimates] = useState({});
  const [qValues, setQValues] = useState({});
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [showFairyTale, setShowFairyTale] = useState(null);
  const [lastAction, setLastAction] = useState(null);
  const [totalReward, setTotalReward] = useState(0);
  const [message, setMessage] = useState("Welcome, brave explorer! Click 'Begin Journey' to watch the agent learn.");

  const goalPos = [GRID_SIZE - 1, GRID_SIZE - 1];
  const obstaclePos = [1, 2];
  const treasurePos = [2, 1];

  const getStateKey = (pos) => `${pos[0]},${pos[1]}`;
  const getSAKey = (pos, action) => `${pos[0]},${pos[1]},${action}`;

  const isValidPos = (pos) => {
    return pos[0] >= 0 && pos[0] < GRID_SIZE && 
           pos[1] >= 0 && pos[1] < GRID_SIZE &&
           !(pos[0] === obstaclePos[0] && pos[1] === obstaclePos[1]);
  };

  const getNextPos = (pos, action) => {
    const delta = ACTION_DELTAS[action];
    const newPos = [pos[0] + delta[0], pos[1] + delta[1]];
    return isValidPos(newPos) ? newPos : pos;
  };

  const getReward = (pos) => {
    if (pos[0] === goalPos[0] && pos[1] === goalPos[1]) return 50;
    if (pos[0] === treasurePos[0] && pos[1] === treasurePos[1]) return 10;
    return -1;
  };

  const isKnown = useCallback((pos, action) => {
    const key = getSAKey(pos, action);
    return (visitCounts[key] || 0) >= KNOWN_THRESHOLD;
  }, [visitCounts]);

  const getEffectiveReward = useCallback((pos, action) => {
    if (isKnown(pos, action)) {
      const key = getSAKey(pos, action);
      return rewardEstimates[key] || 0;
    }
    return RMAX;
  }, [isKnown, rewardEstimates]);

  const computeQValues = useCallback(() => {
    const newQ = {};
    
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        if (i === obstaclePos[0] && j === obstaclePos[1]) continue;
        
        const pos = [i, j];
        for (const action of ACTIONS) {
          const key = getSAKey(pos, action);
          
          if (!isKnown(pos, action)) {
            newQ[key] = RMAX / (1 - GAMMA);
          } else {
            const nextPos = getNextPos(pos, action);
            const reward = rewardEstimates[key] || 0;
            
            let maxNextQ = 0;
            for (const nextAction of ACTIONS) {
              const nextKey = getSAKey(nextPos, nextAction);
              maxNextQ = Math.max(maxNextQ, qValues[nextKey] || 0);
            }
            
            newQ[key] = reward + GAMMA * maxNextQ;
          }
        }
      }
    }
    
    return newQ;
  }, [isKnown, rewardEstimates, qValues]);

  const selectAction = useCallback((pos) => {
    let bestAction = ACTIONS[0];
    let bestQ = -Infinity;
    
    for (const action of ACTIONS) {
      const key = getSAKey(pos, action);
      const q = qValues[key] || (isKnown(pos, action) ? 0 : RMAX / (1 - GAMMA));
      
      if (q > bestQ) {
        bestQ = q;
        bestAction = action;
      }
    }
    
    return bestAction;
  }, [qValues, isKnown]);

  const takeStep = useCallback(() => {
    const action = selectAction(agentPos);
    const nextPos = getNextPos(agentPos, action);
    const reward = getReward(nextPos);
    const saKey = getSAKey(agentPos, action);
    
    const wasKnown = isKnown(agentPos, action);
    
    setVisitCounts(prev => ({
      ...prev,
      [saKey]: (prev[saKey] || 0) + 1
    }));
    
    setRewardEstimates(prev => {
      const oldCount = visitCounts[saKey] || 0;
      const oldEst = prev[saKey] || 0;
      const newEst = (oldEst * oldCount + reward) / (oldCount + 1);
      return { ...prev, [saKey]: newEst };
    });
    
    if (!wasKnown && (visitCounts[saKey] || 0) + 1 >= KNOWN_THRESHOLD) {
      setMessage(`✨ The path ${action} from [${agentPos}] is now KNOWN! No more fairy tales needed here.`);
    } else if (!wasKnown) {
      setShowFairyTale({ pos: agentPos, action });
      setMessage(`🏰 Unknown territory! The agent imagines a fairy tale land with reward ${RMAX}...`);
    } else {
      setMessage(`Walking the known path... Reward: ${reward.toFixed(1)}`);
    }
    
    setLastAction(action);
    setTotalReward(prev => prev + reward);
    setStep(prev => prev + 1);
    
    if (nextPos[0] === goalPos[0] && nextPos[1] === goalPos[1]) {
      setMessage(`🎉 Reached the castle! Episode ${episode + 1} complete. Total reward: ${(totalReward + reward).toFixed(1)}`);
      setAgentPos([0, 0]);
      setEpisode(prev => prev + 1);
      setTotalReward(0);
    } else {
      setAgentPos(nextPos);
    }
    
    setTimeout(() => setShowFairyTale(null), 300);
  }, [agentPos, selectAction, isKnown, visitCounts, episode, totalReward]);

  useEffect(() => {
    const newQ = computeQValues();
    setQValues(newQ);
  }, [visitCounts, rewardEstimates]);

  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(takeStep, speed);
    }
    return () => clearInterval(interval);
  }, [isRunning, takeStep, speed]);

  const reset = () => {
    setAgentPos([0, 0]);
    setVisitCounts({});
    setTransitionCounts({});
    setRewardEstimates({});
    setQValues({});
    setEpisode(0);
    setStep(0);
    setIsRunning(false);
    setTotalReward(0);
    setMessage("Fresh start! All memories erased. The world is full of mystery again.");
  };

  const getVisitCount = (pos, action) => visitCounts[getSAKey(pos, action)] || 0;
  
  const getCellColor = (i, j) => {
    if (i === goalPos[0] && j === goalPos[1]) return 'linear-gradient(135deg, #ffd700 0%, #ff8c00 100%)';
    if (i === obstaclePos[0] && j === obstaclePos[1]) return 'linear-gradient(135deg, #2d3436 0%, #636e72 100%)';
    if (i === treasurePos[0] && j === treasurePos[1]) return 'linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%)';
    
    let totalVisits = 0;
    for (const action of ACTIONS) {
      totalVisits += getVisitCount([i, j], action);
    }
    const alpha = Math.min(totalVisits / 20, 0.6);
    return `rgba(116, 185, 255, ${alpha})`;
  };

  const getMaxQ = (pos) => {
    let maxQ = -Infinity;
    for (const action of ACTIONS) {
      const key = getSAKey(pos, action);
      maxQ = Math.max(maxQ, qValues[key] || 0);
    }
    return maxQ === -Infinity ? 0 : maxQ;
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #0c0c1e 0%, #1a1a3e 50%, #2d2d5a 100%)',
      fontFamily: "'Crimson Text', Georgia, serif",
      color: '#f4e4bc',
      padding: '20px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Stars background */}
      <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
        {[...Array(50)].map((_, i) => (
          <div key={i} style={{
            position: 'absolute',
            width: Math.random() * 3 + 1 + 'px',
            height: Math.random() * 3 + 1 + 'px',
            background: '#fff',
            borderRadius: '50%',
            left: Math.random() * 100 + '%',
            top: Math.random() * 100 + '%',
            animation: `twinkle ${Math.random() * 3 + 2}s infinite`,
            opacity: Math.random() * 0.7 + 0.3
          }} />
        ))}
      </div>
      
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Cinzel:wght@400;700&display=swap');
        @keyframes twinkle {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.5); }
          50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.8); }
        }
        @keyframes fairySparkle {
          0% { transform: scale(0) rotate(0deg); opacity: 1; }
          100% { transform: scale(2) rotate(180deg); opacity: 0; }
        }
      `}</style>

      <div style={{ position: 'relative', zIndex: 1, maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <h1 style={{
            fontFamily: "'Cinzel', serif",
            fontSize: '2.8rem',
            fontWeight: 700,
            background: 'linear-gradient(180deg, #ffd700 0%, #ff8c00 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 30px rgba(255, 215, 0, 0.3)',
            margin: 0
          }}>
            ✦ The Rmax Algorithm ✦
          </h1>
          <p style={{ 
            fontSize: '1.1rem', 
            opacity: 0.9, 
            fontStyle: 'italic',
            marginTop: '10px'
          }}>
            A Tale of Optimistic Exploration
          </p>
        </div>

        {/* Message Banner */}
        <div style={{
          background: 'linear-gradient(90deg, transparent, rgba(244, 228, 188, 0.1), transparent)',
          padding: '15px 30px',
          borderRadius: '10px',
          textAlign: 'center',
          marginBottom: '25px',
          border: '1px solid rgba(244, 228, 188, 0.2)',
          minHeight: '50px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <p style={{ margin: 0, fontSize: '1.1rem' }}>{message}</p>
        </div>

        <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {/* Grid World */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            padding: '25px',
            borderRadius: '20px',
            border: '2px solid rgba(244, 228, 188, 0.3)'
          }}>
            <h2 style={{ 
              fontFamily: "'Cinzel', serif", 
              textAlign: 'center', 
              marginTop: 0,
              fontSize: '1.4rem'
            }}>
              🗺️ The Kingdom
            </h2>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${GRID_SIZE}, 90px)`,
              gap: '4px',
              marginBottom: '20px'
            }}>
              {[...Array(GRID_SIZE)].map((_, i) => (
                [...Array(GRID_SIZE)].map((_, j) => {
                  const isAgent = agentPos[0] === i && agentPos[1] === j;
                  const isGoal = i === goalPos[0] && j === goalPos[1];
                  const isObstacle = i === obstaclePos[0] && j === obstaclePos[1];
                  const isTreasure = i === treasurePos[0] && j === treasurePos[1];
                  const isFairyTale = showFairyTale && 
                    showFairyTale.pos[0] === i && showFairyTale.pos[1] === j;
                  
                  return (
                    <div key={`${i}-${j}`} style={{
                      width: '90px',
                      height: '90px',
                      background: getCellColor(i, j),
                      borderRadius: '12px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                      border: isAgent ? '3px solid #ffd700' : '1px solid rgba(244, 228, 188, 0.2)',
                      animation: isGoal ? 'glow 2s infinite' : 'none',
                      transition: 'all 0.3s ease'
                    }}>
                      {isFairyTale && (
                        <div style={{
                          position: 'absolute',
                          inset: 0,
                          background: 'radial-gradient(circle, rgba(255,215,0,0.4) 0%, transparent 70%)',
                          animation: 'fairySparkle 0.5s ease-out',
                          borderRadius: '12px'
                        }} />
                      )}
                      
                      {isAgent && (
                        <div style={{
                          fontSize: '2rem',
                          animation: 'float 2s infinite',
                          filter: 'drop-shadow(0 0 10px rgba(255,215,0,0.5))'
                        }}>
                          🧙‍♂️
                        </div>
                      )}
                      {isGoal && !isAgent && <div style={{ fontSize: '2rem' }}>🏰</div>}
                      {isObstacle && <div style={{ fontSize: '2rem' }}>🌲</div>}
                      {isTreasure && !isAgent && <div style={{ fontSize: '2rem' }}>💎</div>}
                      
                      {!isObstacle && (
                        <div style={{
                          position: 'absolute',
                          bottom: '4px',
                          fontSize: '0.65rem',
                          opacity: 0.8,
                          background: 'rgba(0,0,0,0.5)',
                          padding: '2px 6px',
                          borderRadius: '4px'
                        }}>
                          V: {getMaxQ([i, j]).toFixed(0)}
                        </div>
                      )}
                    </div>
                  );
                })
              ))}
            </div>

            {/* Legend */}
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '15px',
              justifyContent: 'center',
              fontSize: '0.9rem'
            }}>
              <span>🧙‍♂️ Agent</span>
              <span>🏰 Goal (+50)</span>
              <span>💎 Treasure (+10)</span>
              <span>🌲 Forest (blocked)</span>
            </div>
          </div>

          {/* Right Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', flex: 1, minWidth: '300px', maxWidth: '450px' }}>
            {/* Controls */}
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
              padding: '20px',
              borderRadius: '20px',
              border: '2px solid rgba(244, 228, 188, 0.3)'
            }}>
              <h2 style={{ 
                fontFamily: "'Cinzel', serif", 
                textAlign: 'center', 
                marginTop: 0,
                fontSize: '1.3rem'
              }}>
                ⚔️ Controls
              </h2>
              
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '15px' }}>
                <button onClick={() => setIsRunning(!isRunning)} style={{
                  background: isRunning 
                    ? 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)'
                    : 'linear-gradient(135deg, #27ae60 0%, #1e8449 100%)',
                  border: 'none',
                  color: '#fff',
                  padding: '12px 24px',
                  borderRadius: '25px',
                  cursor: 'pointer',
                  fontFamily: "'Cinzel', serif",
                  fontSize: '1rem',
                  fontWeight: 600,
                  transition: 'transform 0.2s',
                }}>
                  {isRunning ? '⏸ Pause' : '▶ Begin Journey'}
                </button>
                
                <button onClick={takeStep} disabled={isRunning} style={{
                  background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                  border: 'none',
                  color: '#fff',
                  padding: '12px 24px',
                  borderRadius: '25px',
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                  fontFamily: "'Cinzel', serif",
                  fontSize: '1rem',
                  opacity: isRunning ? 0.5 : 1
                }}>
                  👣 One Step
                </button>
                
                <button onClick={reset} style={{
                  background: 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)',
                  border: 'none',
                  color: '#fff',
                  padding: '12px 24px',
                  borderRadius: '25px',
                  cursor: 'pointer',
                  fontFamily: "'Cinzel', serif",
                  fontSize: '1rem'
                }}>
                  🔄 Reset
                </button>
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>
                  Speed: {speed}ms
                </label>
                <input
                  type="range"
                  min="100"
                  max="1000"
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                  style={{ width: '80%', accentColor: '#ffd700' }}
                />
              </div>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '10px',
                marginTop: '15px',
                textAlign: 'center'
              }}>
                <div style={{
                  background: 'rgba(255, 215, 0, 0.1)',
                  padding: '10px',
                  borderRadius: '10px'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ffd700' }}>{episode}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>Episodes</div>
                </div>
                <div style={{
                  background: 'rgba(116, 185, 255, 0.1)',
                  padding: '10px',
                  borderRadius: '10px'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#74b9ff' }}>{step}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>Total Steps</div>
                </div>
                <div style={{
                  background: 'rgba(162, 155, 254, 0.1)',
                  padding: '10px',
                  borderRadius: '10px'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#a29bfe' }}>{totalReward.toFixed(0)}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>Ep. Reward</div>
                </div>
              </div>
            </div>

            {/* State-Action Details */}
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
              padding: '20px',
              borderRadius: '20px',
              border: '2px solid rgba(244, 228, 188, 0.3)'
            }}>
              <h2 style={{ 
                fontFamily: "'Cinzel', serif", 
                textAlign: 'center', 
                marginTop: 0,
                fontSize: '1.3rem'
              }}>
                📜 Knowledge at [{agentPos[0]}, {agentPos[1]}]
              </h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
                {ACTIONS.map(action => {
                  const visits = getVisitCount(agentPos, action);
                  const known = visits >= KNOWN_THRESHOLD;
                  const key = getSAKey(agentPos, action);
                  const q = qValues[key] || (known ? 0 : RMAX / (1 - GAMMA));
                  
                  return (
                    <div key={action} style={{
                      background: known 
                        ? 'rgba(39, 174, 96, 0.2)'
                        : 'rgba(255, 215, 0, 0.15)',
                      padding: '12px',
                      borderRadius: '10px',
                      border: known 
                        ? '1px solid rgba(39, 174, 96, 0.5)'
                        : '1px solid rgba(255, 215, 0, 0.5)',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>
                        {action}
                      </div>
                      <div style={{ fontSize: '0.85rem' }}>
                        Visits: {visits}/{KNOWN_THRESHOLD}
                      </div>
                      <div style={{ 
                        fontSize: '0.85rem',
                        color: known ? '#27ae60' : '#ffd700'
                      }}>
                        {known ? '✓ Known' : '🏰 Fairy Tale'}
                      </div>
                      <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                        Q: {q.toFixed(1)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Fairy Tale Explanation */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(155, 89, 182, 0.2) 0%, rgba(142, 68, 173, 0.2) 100%)',
          padding: '25px',
          borderRadius: '20px',
          marginTop: '25px',
          border: '2px solid rgba(155, 89, 182, 0.4)'
        }}>
          <h2 style={{ 
            fontFamily: "'Cinzel', serif", 
            textAlign: 'center', 
            marginTop: 0,
            color: '#d4a5ff',
            fontSize: '1.4rem'
          }}>
            🏰 The Fairy Tale State Explained
          </h2>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
            gap: '20px',
            fontSize: '0.95rem',
            lineHeight: '1.6'
          }}>
            <div>
              <h3 style={{ color: '#ffd700', marginBottom: '10px' }}>The Optimistic Dreamer</h3>
              <p style={{ margin: 0 }}>
                Rmax is an optimistic algorithm. When the agent doesn't know what lies beyond 
                a path (fewer than <strong>{KNOWN_THRESHOLD} visit</strong>), it imagines a 
                <em> fairy tale land</em> where the reward is the maximum possible: <strong>Rmax = {RMAX}</strong>!
              </p>
            </div>
            
            <div>
              <h3 style={{ color: '#74b9ff', marginBottom: '10px' }}>Why Does This Work?</h3>
              <p style={{ margin: 0 }}>
                By dreaming of treasure in unknown places, the agent is <em>motivated to explore</em>. 
                It will visit every uncertain path hoping to find the fairy tale land, 
                and in doing so, learns the true rewards.
              </p>
            </div>
            
            <div>
              <h3 style={{ color: '#27ae60', marginBottom: '10px' }}>Becoming Known</h3>
              <p style={{ margin: 0 }}>
                After visiting a state-action pair <strong>{KNOWN_THRESHOLD} time</strong>, 
                the agent stops dreaming and uses the <em>real observed rewards</em>. 
                The fairy tale fades, replaced by true knowledge.
              </p>
            </div>
            
            <div>
              <h3 style={{ color: '#e74c3c', marginBottom: '10px' }}>PAC-MDP Guarantee</h3>
              <p style={{ margin: 0 }}>
                Rmax has a theoretical guarantee: it will find a near-optimal policy 
                in polynomial time with high probability. The fairy tale mechanism 
                ensures efficient exploration!
              </p>
            </div>
          </div>
        </div>

        {/* Algorithm Steps */}
        <div style={{
          background: 'rgba(0, 0, 0, 0.3)',
          padding: '25px',
          borderRadius: '20px',
          marginTop: '25px',
          border: '2px solid rgba(244, 228, 188, 0.3)'
        }}>
          <h2 style={{ 
            fontFamily: "'Cinzel', serif", 
            textAlign: 'center', 
            marginTop: 0,
            fontSize: '1.4rem'
          }}>
            📖 The Rmax Algorithm
          </h2>
          
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '15px',
            justifyContent: 'center'
          }}>
            {[
              { num: 1, title: 'Initialize', desc: 'Set all state-action pairs as "unknown"' },
              { num: 2, title: 'Plan', desc: 'Use value iteration with Rmax for unknown pairs' },
              { num: 3, title: 'Act', desc: 'Choose action with highest Q-value' },
              { num: 4, title: 'Learn', desc: 'Update visit counts and reward estimates' },
              { num: 5, title: 'Update', desc: `Mark as "known" after ${KNOWN_THRESHOLD} visits` },
              { num: 6, title: 'Repeat', desc: 'Go back to step 2 with new knowledge' },
            ].map(s => (
              <div key={s.num} style={{
                background: 'rgba(255, 215, 0, 0.1)',
                padding: '15px',
                borderRadius: '15px',
                textAlign: 'center',
                width: '150px',
                border: '1px solid rgba(255, 215, 0, 0.3)'
              }}>
                <div style={{
                  width: '35px',
                  height: '35px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #ffd700 0%, #ff8c00 100%)',
                  color: '#1a1a3e',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 10px',
                  fontWeight: 'bold',
                  fontFamily: "'Cinzel', serif"
                }}>
                  {s.num}
                </div>
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{s.title}</div>
                <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RmaxVisualization;
