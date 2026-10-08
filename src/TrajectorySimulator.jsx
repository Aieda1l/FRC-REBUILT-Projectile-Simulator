import React, {useState, useEffect, useCallback, useMemo} from 'react';
import {simulateTrajectory2D} from './trajectory2d.js';
import Toggle from './Toggle.jsx';
import {createOptimizerClient} from './optimizerClient.js';
import Trajectory3DView from './Trajectory3DView.jsx';
import GameSetupPanel from './GameSetupPanel.jsx';
import {loadLibrary, resolveLibrarySelection} from './gameCatalog.js';
import {createTargetGeometry, targetSideProfile} from './scoringTargets.js';
import {gamePieceWireframe} from './gamePieceGeometry.js';
import {applyCalibrationProfile, parseCalibrationProfile} from './calibration.js';
import AdvancedPhysicsPanel from './AdvancedPhysicsPanel.jsx';

// Physics constants and utilities
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const HUB_STATUS_LABELS = {
    'clean-entry': 'CLEAN ENTRY',
    'rim-collision': 'RIM COLLISION',
    'funnel-collision': 'FUNNEL COLLISION',
    miss: 'MISS',
};

// Backspin estimator for hooded flywheel shooters
// This remains a rough launcher heuristic until shooter exit spin is measured.
const estimateBackspin = (flywheelDia, flywheelRPM, ballDia, compression = 0.5, hoodMaterial = 'foam') => {
    const flywheelRadius = flywheelDia / 2;
    const ballRadius = ballDia / 2;

    // Base efficiency for static hooded shooter
    let baseEfficiency = 0.35;

    // Compression factor (normalized to 0.5")
    const compressionFactor = Math.min(1.5, 0.7 + 0.6 * (compression / 0.5));

    // Material factor
    const materialFactors = {foam: 1.2, rubber: 1.1, polycarbonate: 0.8};
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

// Browser simulation delegates to the canonical 3-D RK4 engine.
const simulateTrajectory = simulateTrajectory2D;

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

// Slider component
const Slider = ({label, value, onChange, min, max, step, unit}) => (
    <div className="mb-3">
        <div className="flex justify-between items-center text-sm mb-1">
            <span className="text-slate-300">{label}</span>
            <div className="flex items-center gap-2">
                <input
                    type="number"
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
                    // Added "appearance-none" just to be safe, and kept the other styles
                    className="w-20 bg-slate-700 text-cyan-400 font-mono text-right rounded px-1 border border-slate-600 focus:outline-none focus:border-indigo-500 appearance-none"
                    step={step}
                />
                <span className="text-slate-500 text-xs w-4">{unit}</span>
            </div>
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

// Result display component
const ResultItem = ({label, value, unit, highlight}) => (
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
    const [azimuth, setAzimuth] = useState(0);
    const [spinRPM, setSpinRPM] = useState(2000);

    // Backspin estimator parameters
    const [flywheelDia, setFlywheelDia] = useState(5.91);
    const [flywheelRPM, setFlywheelRPM] = useState(3500);
    const [ballDia, setBallDia] = useState(5.91);
    const [compression, setCompression] = useState(0.5);
    const [showEstimator, setShowEstimator] = useState(false);

    // Physics toggles
    const [enableDrag, setEnableDrag] = useState(true);
    const [enableMagnus, setEnableMagnus] = useState(true);
    const [showIdeal, setShowIdeal] = useState(true);
    const [showEnvelope, setShowEnvelope] = useState(true);
    const [viewMode, setViewMode] = useState('2d');
    const [playbackIndex, setPlaybackIndex] = useState(0);

    // Advanced calibrated physics
    const [robotVelocity, setRobotVelocity] = useState([0, 0, 0]);
    const [wind, setWind] = useState([0, 0, 0]);
    const [calibrationProfile, setCalibrationProfile] = useState(null);
    const [profileError, setProfileError] = useState('');
    const [robustEnabled, setRobustEnabled] = useState(false);
    const [uncertaintyConfig, setUncertaintyConfig] = useState({
        velocity: {kind: 'normal', mean: 0, sigma: 0.25},
        angleDeg: {kind: 'normal', mean: 0, sigma: 0.5},
        spinRPM: {kind: 'normal', mean: 0, sigma: 100},
        mass: {kind: 'normal', mean: 0, sigma: 0.004},
        robotVelocity: [
            {kind: 'normal', mean: 0, sigma: 0},
            {kind: 'normal', mean: 0, sigma: 0},
            {kind: 'normal', mean: 0, sigma: 0},
        ],
        dragMultiplier: {kind: 'normal', mean: 1, sigma: 0.05, min: 0},
        liftMultiplier: {kind: 'normal', mean: 1, sigma: 0.05, min: 0},
    });
    const [uncertaintySeed, setUncertaintySeed] = useState(2026);
    const [robustSampleCount, setRobustSampleCount] = useState(512);
    const [robustOptimizationResult, setRobustOptimizationResult] = useState(null);

    // Optimizer status
    const [optimizerRunning, setOptimizerRunning] = useState(false);
    const [optimizerProgress, setOptimizerProgress] = useState(null);
    const [optimizerStatus, setOptimizerStatus] = useState('');

    // Error margins
    const [velError, setVelError] = useState(0.5);
    const [angleError, setAngleError] = useState(1.0);

    // All simulations, optimizer workers, and views consume the same active profiles.
    const [gameSelection, setGameSelection] = useState(() => resolveLibrarySelection(loadLibrary()));
    const {piece: gamePiece, target: scoringTarget} = gameSelection;
    const targetX = scoringTarget.x;
    const targetY = scoringTarget.z;
    const hubGeometry = useMemo(() => createTargetGeometry(scoringTarget), [scoringTarget]);
    const mass = gamePiece.mass;
    const radius = gamePiece.diameter / 2;
    const dragCoeff = gamePiece.dragCoeff;
    const liftCoeff = gamePiece.liftCoeff;
    const airDensity = 1.204; // ~20 C, sea level; matches Python default environment
    const gravity = 9.81;

    // Sync with Python backend
    const syncWithPython = async () => {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
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

    // Build params object. Calibration profiles only replace aerodynamic model fields.
    const params = useMemo(() => applyCalibrationProfile({
        launchX, launchY, velocity, angleDeg: angle, azimuthDeg: azimuth, spinRPM,
        mass, radius, dragCoeff, liftCoeff, airDensity, gravity,
        gamePiece,
        enableDrag, enableMagnus,
        targetX,
        targetLateralY: scoringTarget.lateralY,
        target: scoringTarget,
        robotVelocity,
        wind,
    }, calibrationProfile), [
        launchX, launchY, velocity, angle, azimuth, spinRPM, enableDrag, enableMagnus,
        targetX, scoringTarget, gamePiece, mass, radius, dragCoeff, liftCoeff,
        robotVelocity, wind, calibrationProfile,
    ]);

    // Run simulation
    const optimizerClient = useMemo(() => createOptimizerClient({
        onProgress: (progress) => {
            setOptimizerProgress(progress);
        },
        onComplete: (optimization) => {
            setOptimizerRunning(false);
            setOptimizerProgress(null);
            setRobustOptimizationResult(optimization.solution?.robust ?? null);
            if (optimization.solution) {
                setVelocity(Math.round(optimization.solution.velocity * 10) / 10);
                setAngle(Math.round(optimization.solution.angle * 10) / 10);
                if (Number.isFinite(optimization.solution.azimuth)) {
                    setAzimuth(Math.round(optimization.solution.azimuth * 10) / 10);
                }
                setOptimizerStatus(
                    Number.isFinite(optimization.solution.azimuth)
                        ? `Clean entry found (azimuth ${optimization.solution.azimuth.toFixed(1)}°)`
                        : 'Clean entry found'
                );
            } else if (optimization.reason === 'lateral-compensation-infeasible') {
                setOptimizerStatus(
                    'No clean entry: lateral robot velocity exceeds the available horizontal muzzle speed at this elevation. Try Best V + Angle.'
                );
            } else {
                const near = optimization.bestNearMiss?.result?.hubInteraction?.classification;
                setOptimizerStatus(near ? `No clean entry found (best: ${near})` : 'No clean entry found');
            }
        },
        onError: (message) => {
            setOptimizerRunning(false);
            setOptimizerProgress(null);
            setOptimizerStatus(`Optimization error: ${message}`);
        },
    }), []);

    useEffect(() => () => optimizerClient.dispose(), [optimizerClient]);

    const result = useMemo(() => simulateTrajectory(params), [params]);

    // Ideal trajectory (no drag)
    const idealResult = useMemo(() => {
        if (!showIdeal) return null;
        return simulateTrajectory({...params, enableDrag: false, enableMagnus: false, spinRPM: 0});
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
        [launchX, launchY, targetX, targetY, velocity]
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

    const startOptimization = useCallback((mode) => {
        setOptimizerRunning(true);
        setOptimizerProgress({evaluatedCandidates: 0, totalCandidates: mode === 'both' ? 980 : mode === 'angle' ? 160 : 100});
        setOptimizerStatus('');
        const optimizerMode = robustEnabled ? 'robust' : mode;
        const optimizerOptions = robustEnabled ? {
            mode,
            uncertainty: uncertaintyConfig,
            coarseSamples: 64,
            finalSamples: robustSampleCount,
            seed: uncertaintySeed,
        } : undefined;
        optimizerClient.start(optimizerMode, params, optimizerOptions);
    }, [
        optimizerClient, params, robustEnabled, uncertaintyConfig,
        robustSampleCount, uncertaintySeed,
    ]);

    const cancelOptimization = useCallback(() => {
        optimizerClient.cancel();
        setOptimizerRunning(false);
        setOptimizerProgress(null);
        setOptimizerStatus('Optimization cancelled');
    }, [optimizerClient]);

    // Apply heuristic flywheel estimates only after an explicit user action.
    const handleApplyEstimate = useCallback(() => {
        setVelocity(Math.round(estimatedExitVel * 10) / 10);
        setSpinRPM(Math.round(estimatedSpin));
    }, [estimatedExitVel, estimatedSpin]);

    const handleProfileFile = useCallback(async (file) => {
        if (!file) return;
        try {
            const profileText = await file.text();
            const parsedProfile = parseCalibrationProfile(profileText);
            setCalibrationProfile(parsedProfile);
            setProfileError('');
        } catch (error) {
            setProfileError(error instanceof Error ? error.message : String(error));
        }
    }, []);

    const clearCalibrationProfile = useCallback(() => {
        setCalibrationProfile(null);
        setProfileError('');
    }, []);

    // Calculate plot bounds
    const plotBounds = useMemo(() => {
        const allX = result.points.map(p => p.x);
        const allY = result.points.map(p => p.y);
        return {
            xMin: Math.min(launchX - 0.5, targetX - 1.5, ...allX),
            xMax: Math.max(1.5, targetX + 1.5, ...allX) + 0.5,
            yMin: -0.2,
            yMax: Math.max(targetY + 1, result.maxHeight + 0.5)
        };
    }, [result, launchX, targetX, targetY]);

    // Convert coordinates to SVG
    const toSVG = useCallback((x, y) => {
        const {xMin, xMax, yMin, yMax} = plotBounds;
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
            const {x, y} = toSVG(p.x, p.y);
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
    }, [result, toSVG]);

    const idealPath = useMemo(() => {
        if (!idealResult || idealResult.points.length < 2) return '';
        return idealResult.points.map((p, i) => {
            const {x, y} = toSVG(p.x, p.y);
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
    }, [idealResult, toSVG]);

    const envelopePaths = useMemo(() => {
        return envelopeResults.map(r =>
            r.points.map((p, i) => {
                const {x, y} = toSVG(p.x, p.y);
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
            }).join(' ')
        );
    }, [envelopeResults, toSVG]);

    // Side-profile drawing uses the same geometry object as scoring.
    const targetVis = useMemo(() => {
        const side = targetSideProfile(hubGeometry, radius);
        return {
            center: toSVG(...side.labelPoint),
            polygon: side.polygon.map((point) => toSVG(...point)),
            clearances: side.clearances.map((line) => line.map((point) => toSVG(...point))),
        };
    }, [toSVG, hubGeometry, radius]);

    const activeSample = result.samples3d[Math.min(playbackIndex, result.samples3d.length - 1)];
    const pieceVis = useMemo(() => {
        if (!activeSample) return {lines: [], points: []};
        const wire = gamePieceWireframe(gamePiece, activeSample.state.slice(0, 3), activeSample.orientation);
        return {
            lines: wire.lines.map((line) => line.map(([x,,z]) => toSVG(x,z))),
            points: wire.points.map((line) => line.map(([x,,z]) => toSVG(x,z))),
        };
    }, [activeSample, gamePiece, toSVG]);
    const centerNearTarget = () => {
        const crossing = result.hubInteraction?.topCrossing;
        if (!crossing) return;
        let nearest=0, delta=Infinity;
        result.samples3d.forEach((sample,index)=>{
            const d=Math.abs(sample.time-crossing.time);
            if (d<delta) {delta=d; nearest=index;}
        });
        setPlaybackIndex(nearest);
    };
    const clearance = result.hubInteraction?.clearanceMargin;
    const clearanceLabel = Number.isFinite(clearance)
        ? `${clearance >= 0 ? '+' : ''}${(clearance * 100).toFixed(1)} cm`
        : 'N/A';

    const hubClassification = result.hubInteraction?.classification ?? 'miss';
    const hubStatus = HUB_STATUS_LABELS[hubClassification] ?? HUB_STATUS_LABELS.miss;
    const cleanEntry = hubClassification === 'clean-entry';

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
                        Configurable FRC game pieces, targets • Air Drag & Magnus Effect Physics
                    </p>
                </div>

                <div className="grid lg:grid-cols-3 gap-4">
                    {/* Controls Panel */}
                    <div className="lg:col-span-1 space-y-4">
                        <GameSetupPanel onSelectionChange={setGameSelection} disabled={optimizerRunning} />
                        {/* Launch Parameters */}
                        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
                            <h2 className="text-lg font-semibold text-indigo-400 mb-3">Launch Parameters</h2>
                            <Slider label="Distance (X)" value={launchX} onChange={setLaunchX} min={-5} max={-0.5}
                                    step={0.1} unit="m"/>
                            <Slider label="Height (Y)" value={launchY} onChange={setLaunchY} min={0.1} max={2}
                                    step={0.05} unit="m"/>
                            <Slider label="Velocity" value={velocity} onChange={setVelocity} min={5} max={25} step={0.5}
                                    unit="m/s"/>
                            <Slider label="Angle" value={angle} onChange={setAngle} min={10} max={85} step={0.5}
                                    unit="°"/>
                            <Slider label={gamePiece.shape === 'sphere' ? 'Backspin' : 'Axial spin'} value={spinRPM} onChange={setSpinRPM} min={0} max={5000} step={100}
                                    unit="RPM"/>

                            <button
                                onClick={() => startOptimization('angle')}
                                disabled={optimizerRunning}
                                className="w-full mt-3 py-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg font-semibold hover:from-green-400 hover:to-emerald-400 transition-all"
                            >
                                Find Optimal Angle
                            </button>

                            <button
                                onClick={() => startOptimization('velocity')}
                                disabled={optimizerRunning}
                                className="w-full mt-2 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg font-semibold hover:from-blue-400 hover:to-cyan-400 transition-all"
                            >
                                Find Optimal Velocity
                            </button>

                            <button
                                onClick={() => startOptimization('both')}
                                disabled={optimizerRunning}
                                className="w-full mt-2 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-semibold hover:from-purple-400 hover:to-pink-400 transition-all"
                            >
                                Find Best V + Angle
                            </button>

                            {optimizerRunning && (
                                <div className="mt-3 rounded-lg border border-slate-600 bg-slate-900/50 p-3">
                                    <div className="flex justify-between text-xs text-slate-300 mb-2">
                                        <span>Optimizing…</span>
                                        <span>
                                            {optimizerProgress
                                                ? `${Math.min(100, Math.round((optimizerProgress.evaluatedCandidates / Math.max(1, optimizerProgress.totalCandidates)) * 100))}%`
                                                : '0%'}
                                        </span>
                                    </div>
                                    <div className="h-2 rounded bg-slate-700 overflow-hidden">
                                        <div
                                            className="h-full bg-cyan-400 transition-[width]"
                                            style={{
                                                width: optimizerProgress
                                                    ? `${Math.min(100, (optimizerProgress.evaluatedCandidates / Math.max(1, optimizerProgress.totalCandidates)) * 100)}%`
                                                    : '0%',
                                            }}
                                        />
                                    </div>
                                    <button
                                        type="button"
                                        onClick={cancelOptimization}
                                        className="w-full mt-3 py-2 rounded-lg border border-red-500/60 text-red-300 hover:bg-red-500/10"
                                    >
                                        Cancel Optimization
                                    </button>
                                </div>
                            )}
                            {!optimizerRunning && optimizerStatus && (
                                <p className="mt-3 text-xs text-slate-300" role="status">{optimizerStatus}</p>
                            )}
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
                                    <Slider label="Flywheel RPM" value={flywheelRPM} onChange={setFlywheelRPM}
                                            min={1000} max={8000} step={100} unit="RPM"/>
                                    <Slider label="Flywheel Dia" value={flywheelDia} onChange={setFlywheelDia} min={2}
                                            max={8} step={0.5} unit="in"/>
                                    <Slider label="Ball Diameter" value={ballDia} onChange={setBallDia} min={3} max={10}
                                            step={0.5} unit="in"/>
                                    <Slider label="Compression" value={compression} onChange={setCompression} min={0.1}
                                            max={1.5} step={0.1} unit="in"/>

                                    <div className="mt-3 p-3 bg-slate-700/50 rounded-lg">
                                        <div className="text-sm text-slate-300">
                                            Estimated Backspin: <span
                                            className="text-amber-400 font-mono">{estimatedSpin.toFixed(0)} RPM</span>
                                        </div>
                                        <div className="text-sm text-slate-300">
                                            Est. Exit Velocity: <span
                                            className="text-cyan-400 font-mono">{estimatedExitVel.toFixed(1)} m/s</span>
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleApplyEstimate}
                                        className="w-full mt-3 py-2 bg-gradient-to-r from-amber-500 to-orange-500 rounded-lg font-semibold hover:from-amber-400 hover:to-orange-400 transition-all"
                                    >
                                        Apply Estimate
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Physics Options */}
                        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
                            <h2 className="text-lg font-semibold text-indigo-400 mb-3">Physics Options</h2>
                            <Toggle label="Air Drag" checked={enableDrag} onChange={setEnableDrag}/>
                            <Toggle label="Magnus Effect (Backspin)" checked={enableMagnus} onChange={setEnableMagnus}/>
                            <Toggle label="Show Ideal (No Drag)" checked={showIdeal} onChange={setShowIdeal}/>
                            <Toggle label="Show Error Envelope" checked={showEnvelope} onChange={setShowEnvelope}/>

                            {showEnvelope && (
                                <div className="mt-3 pt-3 border-t border-slate-600">
                                    <Slider label="Velocity ±" value={velError} onChange={setVelError} min={0} max={2}
                                            step={0.1} unit="m/s"/>
                                    <Slider label="Angle ±" value={angleError} onChange={setAngleError} min={0} max={5}
                                            step={0.1} unit="°"/>
                                </div>
                            )}
                        </div>

                        <AdvancedPhysicsPanel
                            aimAzimuth={azimuth}
                            onAimAzimuthChange={setAzimuth}
                            robotVelocity={robotVelocity}
                            onRobotVelocityChange={setRobotVelocity}
                            wind={wind}
                            onWindChange={setWind}
                            calibrationProfile={calibrationProfile}
                            profileError={profileError}
                            onProfileFile={handleProfileFile}
                            onClearProfile={clearCalibrationProfile}
                            calibrationDiagnostics={result.calibrationDiagnostics}
                            robustEnabled={robustEnabled}
                            onRobustEnabledChange={setRobustEnabled}
                            uncertaintyConfig={uncertaintyConfig}
                            onUncertaintyConfigChange={setUncertaintyConfig}
                            uncertaintySeed={uncertaintySeed}
                            onUncertaintySeedChange={setUncertaintySeed}
                            robustSampleCount={robustSampleCount}
                            onRobustSampleCountChange={setRobustSampleCount}
                            robustResult={robustOptimizationResult}
                        />

                        {/* Results */}
                        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
                            <h2 className="text-lg font-semibold text-indigo-400 mb-3">Results</h2>
                            <ResultItem
                                label="Hit Target"
                                value={result.hitTarget ? '✓ YES' : '✗ NO'}
                                unit=""
                                highlight={result.hitTarget}
                            />
                            <ResultItem label="Target" value={scoringTarget.name} unit="" />
                            <ResultItem label="Shot points" value={result.hitTarget ? scoringTarget.points ?? 1 : 0} unit="" highlight={result.hitTarget} />
                            <ResultItem label="Game piece" value={gamePiece.name} unit="" />
                            <ResultItem label="Flight Time" value={result.flightTime.toFixed(3)} unit="s"/>
                            <ResultItem label="Max Height" value={result.maxHeight.toFixed(2)} unit="m"/>
                            <ResultItem label="Range" value={result.range.toFixed(2)} unit="m"/>
                            {result.entryVelocity && (
                                <>
                                    <ResultItem label="Entry Velocity" value={result.entryVelocity.toFixed(1)}
                                                unit="m/s"/>
                                    <ResultItem label="Entry Angle" value={result.entryAngle.toFixed(1)} unit="°"/>
                                </>
                            )}
                            <div className="mt-3 pt-3 border-t border-slate-600">
                                <ResultItem label="Ideal Angle (no drag)" value={idealAngle?.toFixed(1) || 'N/A'}
                                            unit="°"/>
                            </div>
                        </div>
                    </div>

                    {/* Visualization */}
                    <div className="lg:col-span-2">
                        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
                            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                                <h2 className="text-lg font-semibold text-indigo-400">Trajectory Graph</h2>
                                <div className="flex items-center gap-2">
                                    <div className="flex rounded-lg border border-slate-600 overflow-hidden" aria-label="Trajectory view">
                                        <button type="button" onClick={() => setViewMode('2d')} className={`px-3 py-1 text-xs ${viewMode === '2d' ? 'bg-indigo-500 text-white' : 'text-slate-300'}`}>2-D</button>
                                        <button type="button" onClick={() => setViewMode('3d')} className={`px-3 py-1 text-xs ${viewMode === '3d' ? 'bg-indigo-500 text-white' : 'text-slate-300'}`}>3-D</button>
                                    </div>
                                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${cleanEntry ? 'bg-green-500/20 text-green-400 border border-green-500/50' : 'bg-red-500/20 text-red-400 border border-red-500/50'}`}>{hubStatus}</span>
                                </div>
                            </div>

                            {viewMode === '3d' ? (
                                <Trajectory3DView
                                    samples={result.samples3d}
                                    idealSamples={showIdeal ? idealResult?.samples3d ?? [] : []}
                                    envelopeSamples={showEnvelope ? envelopeResults.map((entry) => entry.samples3d) : []}
                                    hubGeometry={hubGeometry}
                                    interaction={result.hubInteraction}
                                    ballRadius={radius}
                                    gamePiece={gamePiece}
                                />
                            ) : (
                            <svg viewBox="0 0 600 400" className="w-full h-auto bg-slate-900/50 rounded-lg">
                                {/* Grid */}
                                <defs>
                                    <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                                        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#334155" strokeWidth="0.5"/>
                                    </pattern>
                                </defs>
                                <rect width="600" height="400" fill="url(#grid)"/>

                                {/* Target aperture side profile */}
                                <polyline points={targetVis.polygon.map((p) => p.x + ',' + p.y).join(' ')}
                                          fill="none" stroke="#22c55e" strokeWidth="3"/>
                                {targetVis.clearances.map((line, index) => (
                                    <polyline key={'target-clearance-' + index}
                                              points={line.map((p) => p.x + ',' + p.y).join(' ')}
                                              fill="none" stroke="#67e8f9" strokeWidth="1.5" strokeDasharray="5,5"/>
                                ))}

                                {/* Error envelope */}
                                {envelopePaths.map((path, i) => (
                                    <path key={i} d={path} fill="none" stroke="#f59e0b" strokeWidth="1" opacity="0.3"/>
                                ))}

                                {/* Ideal trajectory */}
                                {idealPath && (
                                    <path d={idealPath} fill="none" stroke="#22d3ee" strokeWidth="2"
                                          strokeDasharray="8,4" opacity="0.7"/>
                                )}

                                {/* Main trajectory */}
                                <path d={trajectoryPath} fill="none" stroke="#818cf8" strokeWidth="3"/>

                                {/* Game-piece outline at selected trajectory sample, scaled in meters */}
                                {pieceVis.lines.map((line,i) => (
                                    <polygon key={'piece-line-'+i} points={line.map(p=>p.x+','+p.y).join(' ')}
                                             stroke="#fbbf24" strokeWidth="1.8"
                                             fill={i===0 ? 'rgba(251,191,36,0.12)' : 'none'}/>
                                ))}
                                {pieceVis.points.map((line,i) => (
                                    <polyline key={'piece-edge-'+i} points={line.map(p=>p.x+','+p.y).join(' ')}
                                              fill="none" stroke="#f59e0b" strokeWidth="1.3"/>
                                ))}

                                {/* Launch point */}
                                <circle cx={launchVis.x} cy={launchVis.y} r="8" fill="#ef4444" stroke="white"
                                        strokeWidth="2"/>

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
                                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5"
                                            orient="auto">
                                        <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444"/>
                                    </marker>
                                </defs>

                                {/* Impact point */}
                                {impactVis && (
                                    <g>
                                        <line x1={impactVis.x - 8} y1={impactVis.y - 8} x2={impactVis.x + 8}
                                              y2={impactVis.y + 8}
                                              stroke={result.hitTarget ? '#22c55e' : '#ef4444'} strokeWidth="3"/>
                                        <line x1={impactVis.x + 8} y1={impactVis.y - 8} x2={impactVis.x - 8}
                                              y2={impactVis.y + 8}
                                              stroke={result.hitTarget ? '#22c55e' : '#ef4444'} strokeWidth="3"/>
                                    </g>
                                )}

                                {/* Legend */}
                                <g transform="translate(450, 20)">
                                    <rect x="0" y="0" width="140" height="90" fill="rgba(15, 23, 42, 0.8)" rx="4"/>
                                    <line x1="10" y1="18" x2="40" y2="18" stroke="#818cf8" strokeWidth="3"/>
                                    <text x="50" y="22" fill="#94a3b8" fontSize="11">Trajectory</text>
                                    <line x1="10" y1="38" x2="40" y2="38" stroke="#22d3ee" strokeWidth="2"
                                          strokeDasharray="5,3"/>
                                    <text x="50" y="42" fill="#94a3b8" fontSize="11">Ideal (no drag)</text>
                                    <line x1="10" y1="58" x2="40" y2="58" stroke="#f59e0b" strokeWidth="1"/>
                                    <text x="50" y="62" fill="#94a3b8" fontSize="11">Error envelope</text>
                                    <line x1="10" y1="78" x2="40" y2="78" stroke="#22c55e" strokeWidth="3"/>
                                    <text x="50" y="82" fill="#94a3b8" fontSize="11">Target</text>
                                </g>

                                {/* Target label */}
                                <text x={targetVis.center.x} y={targetVis.center.y - 30} fill="white" fontSize="12"
                                      textAnchor="middle" fontWeight="bold">
                                    {scoringTarget.name}
                                </text>
                            </svg>
                            )}

                            {viewMode === '2d' && <label className="block mt-3 text-xs text-slate-400">
                                Game-piece position along trajectory
                                <input type="range" className="w-full mt-1 accent-amber-400" min="0"
                                    max={Math.max(0,result.samples3d.length-1)} value={Math.min(playbackIndex,result.samples3d.length-1)}
                                    onChange={(event)=>setPlaybackIndex(Number(event.target.value))}/>
                            </label>}
                            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-300">
                                <button type="button" className="border border-slate-600 px-2 py-1 rounded hover:border-cyan-400"
                                    onClick={centerNearTarget} disabled={!result.hubInteraction?.topCrossing}>
                                    Inspect goal crossing
                                </button>
                                <span>Edge clearance: <strong className={clearance >= 0 ? 'text-green-300' : 'text-amber-300'}>{clearanceLabel}</strong></span>
                                <span className="text-amber-200">Amber = actual-size {gamePiece.shape} ({(gamePiece.diameter*100).toFixed(1)} cm)</span>
                            </div>

                            {/* Info bar */}
                            <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
                                <div className="bg-slate-700/50 rounded p-2">
                                    <div className="text-slate-400">Velocity</div>
                                    <div className="text-cyan-400 font-mono">{(velocity * 3.281).toFixed(1)} ft/s</div>
                                </div>
                                <div className="bg-slate-700/50 rounded p-2">
                                    <div className="text-slate-400">Distance</div>
                                    <div className="text-cyan-400 font-mono">{Math.abs(launchX * 3.281).toFixed(1)} ft
                                    </div>
                                </div>
                                <div className="bg-slate-700/50 rounded p-2">
                                    <div className="text-slate-400">Target Height</div>
                                    <div className="text-cyan-400 font-mono">{(targetY * 3.281).toFixed(1)} ft</div>
                                </div>
                                <div className="bg-slate-700/50 rounded p-2">
                                    <div className="text-slate-400">Game Piece</div>
                                    <div className="text-cyan-400 font-mono">{gamePiece.name}</div>
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
                                            <>Active at <span className="text-cyan-400 font-mono">{spinRPM} RPM</span> -
                                                provides lift</>
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
