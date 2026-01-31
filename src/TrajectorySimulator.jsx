import React, { useState, useEffect, useCallback, useMemo } from 'react';

// Physics constants and utilities
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

// Backspin estimator for hooded flywheel shooters
const estimateBackspin = (flywheelDia, flywheelRPM, ballDia, compression = 0.5, hoodMaterial = 'foam') => {
  const flywheelRadius = flywheelDia / 2;
  const ballRadius = ballDia / 2;
  
  // Base efficiency for static hooded shooter
  let baseEfficiency = 0.35;
  
  // Compression factor (normalized to 0.5")
  const compressionFactor = Math.min(1.5, 0.7 + 0.6 * (compression / 0.5));
  
  // Material factor
  const materialFactors = { foam: 1.2, rubber: 1.1, polycarbonate: 0.8 };
  const materialFactor = materialFactors[hoodMaterial] || 1.0;
  
  // Combined efficiency (capped at realistic max)
  const totalEfficiency = Math.min(0.7, baseEfficiency * compressionFactor * materialFactor);
  
  // Backspin = flywheel_rpm * (r_flywheel/r_ball) * efficiency
  return flywheelRPM * (flywheelRadius / ballRadius) * totalEfficiency;
};

// Estimate exit velocity from flywheel
const estimateExitVelocity = (flywheelDia, flywheelRPM) => {
  const flywheelRadiusM = (flywheelDia / 2) * 0.0254; // to meters
  const surfaceSpeed = flywheelRPM * 2 * Math.PI * flywheelRadiusM / 60;
  return surfaceSpeed * 0.55; // ~55% efficiency for single flywheel
};

// Physics simulation with air drag and Magnus effect
const simulateTrajectory = (params) => {
  const {
    launchX, launchY, velocity, angleDeg, spinRPM,
    mass, radius, dragCoeff, liftCoeff,
    airDensity, gravity,
    enableDrag, enableMagnus,
    targetX, targetY, targetRadius
  } = params;

  const dt = 0.001;
  const maxTime = 5.0;
  
  const angleRad = angleDeg * DEG_TO_RAD;
  let x = launchX;
  let y = launchY;
  let vx = velocity * Math.cos(angleRad);
  let vy = velocity * Math.sin(angleRad);
  let spin = spinRPM * 2 * Math.PI / 60; // Convert to rad/s
  let t = 0;
  
  const crossSection = Math.PI * radius * radius;
  const dragFactor = enableDrag ? 0.5 * airDensity * crossSection * dragCoeff : 0;
  const magnusFactor = enableMagnus ? 0.5 * airDensity * crossSection * liftCoeff : 0;
  
  const points = [{ t, x, y, vx, vy, speed: velocity }];
  let maxHeight = y;
  let hitTarget = false;
  let impactPoint = null;
  let entryVelocity = null;
  let entryAngle = null;
  
  while (t < maxTime && y >= 0) {
    const speed = Math.sqrt(vx * vx + vy * vy);
    
    // Acceleration from gravity
    let ax = 0;
    let ay = -gravity;
    
    // Air drag
    if (speed > 0.001 && dragFactor > 0) {
      const dragMag = dragFactor * speed * speed;
      ax -= (dragMag * vx / speed) / mass;
      ay -= (dragMag * vy / speed) / mass;
    }
    
    // Magnus effect (backspin creates lift)
    if (speed > 0.001 && Math.abs(spin) > 0.001 && magnusFactor > 0) {
      const spinParam = Math.abs(spin) * radius / speed;
      const effectiveCl = liftCoeff * Math.min(spinParam, 0.5) * 2;
      const magnusMag = magnusFactor * speed * speed * effectiveCl;
      const sign = Math.sign(spin);
      ax += (-sign * magnusMag * vy / speed) / mass;
      ay += (sign * magnusMag * vx / speed) / mass;
    }
    
    // RK4-style integration (simplified)
    x += vx * dt + 0.5 * ax * dt * dt;
    y += vy * dt + 0.5 * ay * dt * dt;
    vx += ax * dt;
    vy += ay * dt;
    spin *= Math.exp(-dt / 3.0); // Spin decay
    t += dt;
    
    maxHeight = Math.max(maxHeight, y);
    
    // Check target intersection
    if (y <= targetY && points.length > 1 && points[points.length - 1].y > targetY) {
      const prev = points[points.length - 1];
      const alpha = (targetY - prev.y) / (y - prev.y);
      const crossX = prev.x + alpha * (x - prev.x);
      const crossVx = prev.vx + alpha * (vx - prev.vx);
      const crossVy = prev.vy + alpha * (vy - prev.vy);
      
      if (Math.abs(crossX - targetX) <= targetRadius) {
        hitTarget = true;
        impactPoint = { x: crossX, y: targetY };
        entryVelocity = Math.sqrt(crossVx * crossVx + crossVy * crossVy);
        entryAngle = Math.atan2(crossVy, crossVx) * RAD_TO_DEG;
      }
    }
    
    // Sample points (every 5ms for display)
    if (Math.floor(t * 200) > Math.floor((t - dt) * 200)) {
      points.push({ t, x, y, vx, vy, speed: Math.sqrt(vx * vx + vy * vy) });
    }
  }
  
  if (!impactPoint) {
    impactPoint = { x, y: Math.max(0, y) };
  }
  
  return {
    points,
    hitTarget,
    impactPoint,
    flightTime: t,
    maxHeight,
    range: x - launchX,
    entryVelocity,
    entryAngle
  };
};

// Compute ideal (no drag) angle analytically
const computeIdealAngle = (launchX, launchY, targetX, targetY, velocity, gravity) => {
  const dx = targetX - launchX;
  const dy = targetY - launchY;
  const v2 = velocity * velocity;
  const v4 = v2 * v2;
  const g = gravity;
  
  const discriminant = v4 - g * (g * dx * dx + 2 * dy * v2);
  if (discriminant < 0) return null;
  
  const sqrtDisc = Math.sqrt(discriminant);
  const angle1 = Math.atan2(v2 + sqrtDisc, g * dx) * RAD_TO_DEG;
  const angle2 = Math.atan2(v2 - sqrtDisc, g * dx) * RAD_TO_DEG;
  
  // Return the lower angle (more practical for shooters)
  return angle1 < angle2 ? angle1 : angle2;
};

// Find optimal angle with drag
const findOptimalAngle = (params) => {
  let bestAngle = null;
  let bestError = Infinity;
  
  for (let angle = 5; angle <= 85; angle += 0.5) {
    const result = simulateTrajectory({ ...params, angleDeg: angle });
    if (result.impactPoint) {
      const error = Math.abs(result.impactPoint.x - params.targetX);
      if (result.impactPoint.y >= params.targetY - 0.1 && error < bestError) {
        bestError = error;
        bestAngle = angle;
      }
    }
  }
  
  // Refine with finer search
  if (bestAngle !== null) {
    for (let angle = bestAngle - 2; angle <= bestAngle + 2; angle += 0.1) {
      const result = simulateTrajectory({ ...params, angleDeg: angle });
      if (result.impactPoint) {
        const error = Math.abs(result.impactPoint.x - params.targetX);
        if (result.hitTarget && error < bestError) {
          bestError = error;
          bestAngle = angle;
        }
      }
    }
  }
  
  return bestAngle;
};

// Find optimal velocity for a given angle
const findOptimalVelocity = (params, minV = 5, maxV = 25) => {
  let bestVel = null;
  let bestError = Infinity;
  
  // Coarse search
  for (let v = minV; v <= maxV; v += 0.5) {
    const result = simulateTrajectory({ ...params, velocity: v });
    if (result.hitTarget) {
      const error = Math.abs(result.impactPoint.x - params.targetX);
      if (error < bestError) {
        bestError = error;
        bestVel = v;
      }
    }
  }
  
  // Refine
  if (bestVel !== null) {
    for (let v = bestVel - 1; v <= bestVel + 1; v += 0.1) {
      const result = simulateTrajectory({ ...params, velocity: v });
      if (result.hitTarget) {
        const error = Math.abs(result.impactPoint.x - params.targetX);
        if (error < bestError) {
          bestError = error;
          bestVel = v;
        }
      }
    }
  }
  
  return bestVel;
};

// Find optimal velocity AND angle
const findOptimalBoth = (params, minV = 5, maxV = 25) => {
  let best = null;
  let bestError = Infinity;
  
  for (let v = minV; v <= maxV; v += 0.5) {
    for (let a = 20; a <= 80; a += 1) {
      const result = simulateTrajectory({ ...params, velocity: v, angleDeg: a });
      if (result.hitTarget) {
        const error = Math.abs(result.impactPoint.x - params.targetX);
        if (error < bestError) {
          bestError = error;
          best = { velocity: v, angle: a, result };
        }
      }
    }
  }
  
  return best;
};

// Slider component
const Slider = ({ label, value, onChange, min, max, step, unit }) => (
  <div className="mb-3">
    <div className="flex justify-between text-sm mb-1">
      <span className="text-slate-300">{label}</span>
      <span className="text-cyan-400 font-mono">{value.toFixed(step < 1 ? 1 : 0)} {unit}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
    />
  </div>
);

// Toggle component
const Toggle = ({ label, checked, onChange }) => (
  <label className="flex items-center gap-2 cursor-pointer mb-2">
    <div className={`w-10 h-5 rounded-full transition-colors ${checked ? 'bg-indigo-500' : 'bg-slate-600'}`}>
      <div className={`w-4 h-4 bg-white rounded-full transform transition-transform mt-0.5 ${checked ? 'translate-x-5 ml-0.5' : 'translate-x-0.5'}`} />
    </div>
    <span className="text-sm text-slate-300">{label}</span>
  </label>
);

// Result display component
const ResultItem = ({ label, value, unit, highlight }) => (
  <div className="flex justify-between py-1 border-b border-slate-700/50">
    <span className="text-slate-400 text-sm">{label}</span>
    <span className={`font-mono text-sm ${highlight ? 'text-green-400' : 'text-cyan-400'}`}>
      {value} {unit}
    </span>
  </div>
);

// Main App
export default function TrajectorySimulator() {
  // Launch parameters
  const [launchX, setLaunchX] = useState(-3.0);
  const [launchY, setLaunchY] = useState(0.5);
  const [velocity, setVelocity] = useState(12.0);
  const [angle, setAngle] = useState(55);
  const [spinRPM, setSpinRPM] = useState(2000);
  
  // Backspin estimator parameters
  const [flywheelDia, setFlywheelDia] = useState(4.0);
  const [flywheelRPM, setFlywheelRPM] = useState(3500);
  const [ballDia, setBallDia] = useState(5.0);
  const [compression, setCompression] = useState(0.5);
  const [showEstimator, setShowEstimator] = useState(false);
  
  // Physics toggles
  const [enableDrag, setEnableDrag] = useState(true);
  const [enableMagnus, setEnableMagnus] = useState(true);
  const [showIdeal, setShowIdeal] = useState(true);
  const [showEnvelope, setShowEnvelope] = useState(true);
  
  // Error margins
  const [velError, setVelError] = useState(0.5);
  const [angleError, setAngleError] = useState(1.0);
  
  // Target (FRC 2026 Hub estimated)
  const targetX = 0;
  const targetY = 2.64;
  const targetRadius = 0.61;
  const funnelRadius = 1.07;
  
  // Game piece (Coral 2026)
  const mass = 0.235;
  const radius = 0.1016;
  const dragCoeff = 0.47;
  const liftCoeff = 0.25;
  const airDensity = 1.225;
  const gravity = 9.81;

  // Sync with Python backend
  const syncWithPython = async () => {
    const response = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        velocity: velocity,
        angle: angle,
        spin_rate: (spinRPM * 2 * Math.PI) / 60,
        launch_x: launchX,
        launch_y: launchY
      }),
    });
    const data = await response.json();
    console.log("Python Result:", data);
  };
  
  // Build params object
  const params = useMemo(() => ({
    launchX, launchY, velocity, angleDeg: angle, spinRPM,
    mass, radius, dragCoeff, liftCoeff, airDensity, gravity,
    enableDrag, enableMagnus,
    targetX, targetY, targetRadius
  }), [launchX, launchY, velocity, angle, spinRPM, enableDrag, enableMagnus]);
  
  // Run simulation
  const result = useMemo(() => simulateTrajectory(params), [params]);
  
  // Ideal trajectory (no drag)
  const idealResult = useMemo(() => {
    if (!showIdeal) return null;
    return simulateTrajectory({ ...params, enableDrag: false, enableMagnus: false, spinRPM: 0 });
  }, [params, showIdeal]);
  
  // Error envelope trajectories
  const envelopeResults = useMemo(() => {
    if (!showEnvelope) return [];
    const results = [];
    const vErrors = [-velError, velError];
    const aErrors = [-angleError, angleError];
    
    for (const dv of vErrors) {
      for (const da of aErrors) {
        results.push(simulateTrajectory({
          ...params,
          velocity: velocity + dv,
          angleDeg: angle + da
        }));
      }
    }
    return results;
  }, [params, showEnvelope, velError, angleError, velocity, angle]);
  
  // Ideal angle calculation
  const idealAngle = useMemo(() => 
    computeIdealAngle(launchX, launchY, targetX, targetY, velocity, gravity),
    [launchX, launchY, velocity]
  );
  
  // Estimated backspin from flywheel params
  const estimatedSpin = useMemo(() => 
    estimateBackspin(flywheelDia, flywheelRPM, ballDia, compression),
    [flywheelDia, flywheelRPM, ballDia, compression]
  );
  
  const estimatedExitVel = useMemo(() =>
    estimateExitVelocity(flywheelDia, flywheelRPM),
    [flywheelDia, flywheelRPM]
  );
  
  // Find optimal handler
  const handleFindOptimalAngle = useCallback(() => {
    const optimal = findOptimalAngle(params);
    if (optimal !== null) {
      setAngle(Math.round(optimal * 10) / 10);
    }
  }, [params]);
  
  // Find optimal velocity handler
  const handleFindOptimalVelocity = useCallback(() => {
    const optimal = findOptimalVelocity(params);
    if (optimal !== null) {
      setVelocity(Math.round(optimal * 10) / 10);
    }
  }, [params]);
  
  // Find optimal both handler
  const handleFindOptimalBoth = useCallback(() => {
    const optimal = findOptimalBoth(params);
    if (optimal !== null) {
      setVelocity(Math.round(optimal.velocity * 10) / 10);
      setAngle(Math.round(optimal.angle * 10) / 10);
    }
  }, [params]);
  
  // Apply estimated backspin
  const handleApplyEstimatedSpin = useCallback(() => {
    setSpinRPM(Math.round(estimatedSpin));
  }, [estimatedSpin]);
  
  // Calculate plot bounds
  const plotBounds = useMemo(() => {
    const allX = result.points.map(p => p.x);
    const allY = result.points.map(p => p.y);
    return {
      xMin: Math.min(launchX - 0.5, ...allX),
      xMax: Math.max(1.5, ...allX) + 0.5,
      yMin: -0.2,
      yMax: Math.max(targetY + 1, result.maxHeight + 0.5)
    };
  }, [result, launchX, targetY]);
  
  // Convert coordinates to SVG
  const toSVG = useCallback((x, y) => {
    const { xMin, xMax, yMin, yMax } = plotBounds;
    const svgWidth = 600;
    const svgHeight = 400;
    const padding = 40;
    
    const scaleX = (svgWidth - 2 * padding) / (xMax - xMin);
    const scaleY = (svgHeight - 2 * padding) / (yMax - yMin);
    const scale = Math.min(scaleX, scaleY);
    
    return {
      x: padding + (x - xMin) * scale,
      y: svgHeight - padding - (y - yMin) * scale
    };
  }, [plotBounds]);
  
  // Generate path
  const trajectoryPath = useMemo(() => {
    if (result.points.length < 2) return '';
    return result.points.map((p, i) => {
      const { x, y } = toSVG(p.x, p.y);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  }, [result, toSVG]);
  
  const idealPath = useMemo(() => {
    if (!idealResult || idealResult.points.length < 2) return '';
    return idealResult.points.map((p, i) => {
      const { x, y } = toSVG(p.x, p.y);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  }, [idealResult, toSVG]);
  
  const envelopePaths = useMemo(() => {
    return envelopeResults.map(r => 
      r.points.map((p, i) => {
        const { x, y } = toSVG(p.x, p.y);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }).join(' ')
    );
  }, [envelopeResults, toSVG]);
  
  // Target visualization points
  const targetVis = useMemo(() => {
    const center = toSVG(targetX, targetY);
    const funnelLeft = toSVG(targetX - funnelRadius, targetY + 0.25);
    const funnelRight = toSVG(targetX + funnelRadius, targetY + 0.25);
    const entryLeft = toSVG(targetX - targetRadius, targetY);
    const entryRight = toSVG(targetX + targetRadius, targetY);
    return { center, funnelLeft, funnelRight, entryLeft, entryRight };
  }, [toSVG]);
  
  const launchVis = useMemo(() => toSVG(launchX, launchY), [toSVG, launchX, launchY]);
  const impactVis = useMemo(() => result.impactPoint ? toSVG(result.impactPoint.x, result.impactPoint.y) : null, [toSVG, result]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
            FRC Trajectory Simulator
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            2026 Season • Air Drag & Magnus Effect Physics
          </p>
        </div>
        
        <div className="grid lg:grid-cols-3 gap-4">
          {/* Controls Panel */}
          <div className="lg:col-span-1 space-y-4">
            {/* Launch Parameters */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-semibold text-indigo-400 mb-3">Launch Parameters</h2>
              <Slider label="Distance (X)" value={launchX} onChange={setLaunchX} min={-5} max={-0.5} step={0.1} unit="m" />
              <Slider label="Height (Y)" value={launchY} onChange={setLaunchY} min={0.1} max={2} step={0.05} unit="m" />
              <Slider label="Velocity" value={velocity} onChange={setVelocity} min={5} max={25} step={0.5} unit="m/s" />
              <Slider label="Angle" value={angle} onChange={setAngle} min={10} max={85} step={0.5} unit="°" />
              <Slider label="Backspin" value={spinRPM} onChange={setSpinRPM} min={0} max={5000} step={100} unit="RPM" />
              
              <button
                onClick={handleFindOptimalAngle}
                className="w-full mt-3 py-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg font-semibold hover:from-green-400 hover:to-emerald-400 transition-all"
              >
                Find Optimal Angle
              </button>
              
              <button
                onClick={handleFindOptimalVelocity}
                className="w-full mt-2 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg font-semibold hover:from-blue-400 hover:to-cyan-400 transition-all"
              >
                Find Optimal Velocity
              </button>
              
              <button
                onClick={handleFindOptimalBoth}
                className="w-full mt-2 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-semibold hover:from-purple-400 hover:to-pink-400 transition-all"
              >
                Find Best V + Angle
              </button>
            </div>
            
            {/* Backspin Estimator */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <button
                onClick={() => setShowEstimator(!showEstimator)}
                className="w-full flex justify-between items-center text-lg font-semibold text-amber-400"
              >
                <span>Backspin Calculator</span>
                <span>{showEstimator ? '▼' : '▶'}</span>
              </button>
              
              {showEstimator && (
                <div className="mt-3 pt-3 border-t border-slate-600">
                  <p className="text-xs text-slate-400 mb-3">
                    For hooded single-flywheel shooters (like 254's 2017)
                  </p>
                  <Slider label="Flywheel RPM" value={flywheelRPM} onChange={setFlywheelRPM} min={1000} max={6000} step={100} unit="RPM" />
                  <Slider label="Flywheel Dia" value={flywheelDia} onChange={setFlywheelDia} min={2} max={8} step={0.5} unit="in" />
                  <Slider label="Ball Diameter" value={ballDia} onChange={setBallDia} min={3} max={10} step={0.5} unit="in" />
                  <Slider label="Compression" value={compression} onChange={setCompression} min={0.1} max={1.5} step={0.1} unit="in" />
                  
                  <div className="mt-3 p-3 bg-slate-700/50 rounded-lg">
                    <div className="text-sm text-slate-300">
                      Estimated Backspin: <span className="text-amber-400 font-mono">{estimatedSpin.toFixed(0)} RPM</span>
                    </div>
                    <div className="text-sm text-slate-300">
                      Est. Exit Velocity: <span className="text-cyan-400 font-mono">{estimatedExitVel.toFixed(1)} m/s</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleApplyEstimatedSpin}
                    className="w-full mt-3 py-2 bg-gradient-to-r from-amber-500 to-orange-500 rounded-lg font-semibold hover:from-amber-400 hover:to-orange-400 transition-all"
                  >
                    Apply Estimated Backspin
                  </button>
                </div>
              )}
            </div>
            
            {/* Physics Options */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-semibold text-indigo-400 mb-3">Physics Options</h2>
              <Toggle label="Air Drag" checked={enableDrag} onChange={setEnableDrag} />
              <Toggle label="Magnus Effect (Backspin)" checked={enableMagnus} onChange={setEnableMagnus} />
              <Toggle label="Show Ideal (No Drag)" checked={showIdeal} onChange={setShowIdeal} />
              <Toggle label="Show Error Envelope" checked={showEnvelope} onChange={setShowEnvelope} />
              
              {showEnvelope && (
                <div className="mt-3 pt-3 border-t border-slate-600">
                  <Slider label="Velocity ±" value={velError} onChange={setVelError} min={0} max={2} step={0.1} unit="m/s" />
                  <Slider label="Angle ±" value={angleError} onChange={setAngleError} min={0} max={5} step={0.1} unit="°" />
                </div>
              )}
            </div>
            
            {/* Results */}
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-semibold text-indigo-400 mb-3">Results</h2>
              <ResultItem 
                label="Hit Target" 
                value={result.hitTarget ? '✓ YES' : '✗ NO'} 
                unit="" 
                highlight={result.hitTarget}
              />
              <ResultItem label="Flight Time" value={result.flightTime.toFixed(3)} unit="s" />
              <ResultItem label="Max Height" value={result.maxHeight.toFixed(2)} unit="m" />
              <ResultItem label="Range" value={result.range.toFixed(2)} unit="m" />
              {result.entryVelocity && (
                <>
                  <ResultItem label="Entry Velocity" value={result.entryVelocity.toFixed(1)} unit="m/s" />
                  <ResultItem label="Entry Angle" value={result.entryAngle.toFixed(1)} unit="°" />
                </>
              )}
              <div className="mt-3 pt-3 border-t border-slate-600">
                <ResultItem label="Ideal Angle (no drag)" value={idealAngle?.toFixed(1) || 'N/A'} unit="°" />
              </div>
            </div>
          </div>
          
          {/* Visualization */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold text-indigo-400">Trajectory Graph</h2>
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  result.hitTarget 
                    ? 'bg-green-500/20 text-green-400 border border-green-500/50' 
                    : 'bg-red-500/20 text-red-400 border border-red-500/50'
                }`}>
                  {result.hitTarget ? '✓ TARGET HIT' : '✗ MISS'}
                </span>
              </div>
              
              <svg viewBox="0 0 600 400" className="w-full h-auto bg-slate-900/50 rounded-lg">
                {/* Grid */}
                <defs>
                  <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                    <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#334155" strokeWidth="0.5" />
                  </pattern>
                </defs>
                <rect width="600" height="400" fill="url(#grid)" />
                
                {/* Target funnel */}
                <path
                  d={`M ${targetVis.funnelLeft.x} ${targetVis.funnelLeft.y} 
                      L ${targetVis.entryLeft.x} ${targetVis.entryLeft.y}
                      L ${targetVis.entryRight.x} ${targetVis.entryRight.y}
                      L ${targetVis.funnelRight.x} ${targetVis.funnelRight.y}`}
                  fill="rgba(34, 197, 94, 0.1)"
                  stroke="#22c55e"
                  strokeWidth="3"
                />
                <line
                  x1={targetVis.entryLeft.x}
                  y1={targetVis.entryLeft.y}
                  x2={targetVis.entryRight.x}
                  y2={targetVis.entryRight.y}
                  stroke="#22c55e"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
                
                {/* Error envelope */}
                {envelopePaths.map((path, i) => (
                  <path key={i} d={path} fill="none" stroke="#f59e0b" strokeWidth="1" opacity="0.3" />
                ))}
                
                {/* Ideal trajectory */}
                {idealPath && (
                  <path d={idealPath} fill="none" stroke="#22d3ee" strokeWidth="2" strokeDasharray="8,4" opacity="0.7" />
                )}
                
                {/* Main trajectory */}
                <path d={trajectoryPath} fill="none" stroke="#818cf8" strokeWidth="3" />
                
                {/* Launch point */}
                <circle cx={launchVis.x} cy={launchVis.y} r="8" fill="#ef4444" stroke="white" strokeWidth="2" />
                
                {/* Velocity arrow */}
                <line
                  x1={launchVis.x}
                  y1={launchVis.y}
                  x2={launchVis.x + Math.cos(angle * DEG_TO_RAD) * 40}
                  y2={launchVis.y - Math.sin(angle * DEG_TO_RAD) * 40}
                  stroke="#ef4444"
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
                  </marker>
                </defs>
                
                {/* Impact point */}
                {impactVis && (
                  <g>
                    <line x1={impactVis.x - 8} y1={impactVis.y - 8} x2={impactVis.x + 8} y2={impactVis.y + 8} 
                          stroke={result.hitTarget ? '#22c55e' : '#ef4444'} strokeWidth="3" />
                    <line x1={impactVis.x + 8} y1={impactVis.y - 8} x2={impactVis.x - 8} y2={impactVis.y + 8} 
                          stroke={result.hitTarget ? '#22c55e' : '#ef4444'} strokeWidth="3" />
                  </g>
                )}
                
                {/* Legend */}
                <g transform="translate(450, 20)">
                  <rect x="0" y="0" width="140" height="90" fill="rgba(15, 23, 42, 0.8)" rx="4" />
                  <line x1="10" y1="18" x2="40" y2="18" stroke="#818cf8" strokeWidth="3" />
                  <text x="50" y="22" fill="#94a3b8" fontSize="11">Trajectory</text>
                  <line x1="10" y1="38" x2="40" y2="38" stroke="#22d3ee" strokeWidth="2" strokeDasharray="5,3" />
                  <text x="50" y="42" fill="#94a3b8" fontSize="11">Ideal (no drag)</text>
                  <line x1="10" y1="58" x2="40" y2="58" stroke="#f59e0b" strokeWidth="1" />
                  <text x="50" y="62" fill="#94a3b8" fontSize="11">Error envelope</text>
                  <line x1="10" y1="78" x2="40" y2="78" stroke="#22c55e" strokeWidth="3" />
                  <text x="50" y="82" fill="#94a3b8" fontSize="11">Target</text>
                </g>
                
                {/* Target label */}
                <text x={targetVis.center.x} y={targetVis.center.y - 30} fill="white" fontSize="12" textAnchor="middle" fontWeight="bold">
                  Hub
                </text>
              </svg>
              
              {/* Info bar */}
              <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Velocity</div>
                  <div className="text-cyan-400 font-mono">{(velocity * 3.281).toFixed(1)} ft/s</div>
                </div>
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Distance</div>
                  <div className="text-cyan-400 font-mono">{Math.abs(launchX * 3.281).toFixed(1)} ft</div>
                </div>
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Target Height</div>
                  <div className="text-cyan-400 font-mono">{(targetY * 3.281).toFixed(1)} ft</div>
                </div>
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Game Piece</div>
                  <div className="text-cyan-400 font-mono">Coral 2026</div>
                </div>
              </div>
            </div>
            
            {/* Physics info */}
            <div className="mt-4 bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-semibold text-indigo-400 mb-2">Physics Details</h2>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-slate-400 mb-1">Air Drag Effect</div>
                  <div className="text-slate-300">
                    {enableDrag ? (
                      idealResult ? (
                        <>Reduces range by <span className="text-amber-400 font-mono">
                          {(idealResult.range - result.range).toFixed(2)}m
                        </span> ({((1 - result.range / idealResult.range) * 100).toFixed(1)}%)</>
                      ) : 'Active - computing...'
                    ) : 'Disabled'}
                  </div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Magnus Effect (Backspin)</div>
                  <div className="text-slate-300">
                    {enableMagnus && spinRPM > 0 ? (
                      <>Active at <span className="text-cyan-400 font-mono">{spinRPM} RPM</span> - provides lift</>
                    ) : 'Disabled or zero spin'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="text-center mt-6 text-slate-500 text-sm">
          FRC Trajectory Simulator • Realistic physics for shooter calibration
        </div>
      </div>
    </div>
  );
}
