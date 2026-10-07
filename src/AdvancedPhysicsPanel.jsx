import React, {useState} from 'react';

function NumberField({label, value, onChange, step = 0.1, min, max, unit = ''}) {
    return (
        <label className="block text-xs text-slate-300">
            <span className="flex justify-between gap-2 mb-1">
                <span>{label}</span>
                {unit && <span className="text-slate-500">{unit}</span>}
            </span>
            <input
                type="number"
                value={value}
                step={step}
                min={min}
                max={max}
                onChange={(event) => onChange(Number(event.target.value))}
                className="w-full rounded border border-slate-600 bg-slate-900/60 px-2 py-1 text-cyan-300"
            />
        </label>
    );
}

function probability(value) {
    return Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : 'N/A';
}

export default function AdvancedPhysicsPanel({
    robotVelocity,
    onRobotVelocityChange,
    wind,
    onWindChange,
    calibrationProfile,
    profileError,
    onProfileFile,
    onClearProfile,
    calibrationDiagnostics,
    robustEnabled,
    onRobustEnabledChange,
    uncertaintyConfig,
    onUncertaintyConfigChange,
    uncertaintySeed,
    onUncertaintySeedChange,
    robustSampleCount,
    onRobustSampleCountChange,
    robustResult,
}) {
    const [open, setOpen] = useState(false);

    const setVector = (values, setter, index, value) => {
        const next = [...values];
        next[index] = Number.isFinite(value) ? value : 0;
        setter(next);
    };
    const setSigma = (key, sigma) => {
        onUncertaintyConfigChange({
            ...uncertaintyConfig,
            [key]: {
                ...uncertaintyConfig[key],
                sigma: Math.max(0, Number.isFinite(sigma) ? sigma : 0),
            },
        });
    };
    const setVectorSigma = (key, index, sigma) => {
        const current = uncertaintyConfig[key] ?? [
            {kind: 'normal', mean: 0, sigma: 0},
            {kind: 'normal', mean: 0, sigma: 0},
            {kind: 'normal', mean: 0, sigma: 0},
        ];
        const next = current.map((distribution, distributionIndex) => (
            distributionIndex === index
                ? {...distribution, sigma: Math.max(0, Number.isFinite(sigma) ? sigma : 0)}
                : {...distribution}
        ));
        onUncertaintyConfigChange({
            ...uncertaintyConfig,
            [key]: next,
        });
    };

    const validationRms = calibrationProfile?.validation?.rms3d;
    const clampedFraction = calibrationDiagnostics?.clampedFraction ?? 0;

    return (
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 border border-slate-700">
            <button
                type="button"
                onClick={() => setOpen((value) => !value)}
                className="w-full flex items-center justify-between text-left"
                aria-expanded={open}
            >
                <span className="text-lg font-semibold text-indigo-400">Advanced Physics</span>
                <span className="text-slate-400">{open ? '▼' : '▶'}</span>
            </button>

            {open && (
                <div className="mt-4 space-y-5 border-t border-slate-700 pt-4">
                    <section>
                        <h3 className="text-sm font-semibold text-slate-200 mb-2">Motion & Environment</h3>
                        <div className="grid grid-cols-2 gap-2">
                            <NumberField label="Robot Forward Velocity" value={robotVelocity[0]}
                                         onChange={(value) => setVector(robotVelocity, onRobotVelocityChange, 0, value)}
                                         step={0.1} unit="m/s"/>
                            <NumberField label="Robot Lateral Velocity" value={robotVelocity[1]}
                                         onChange={(value) => setVector(robotVelocity, onRobotVelocityChange, 1, value)}
                                         step={0.1} unit="m/s"/>
                            <NumberField label="Wind Forward" value={wind[0]}
                                         onChange={(value) => setVector(wind, onWindChange, 0, value)}
                                         step={0.1} unit="m/s"/>
                            <NumberField label="Wind Lateral" value={wind[1]}
                                         onChange={(value) => setVector(wind, onWindChange, 1, value)}
                                         step={0.1} unit="m/s"/>
                        </div>
                    </section>

                    <section>
                        <h3 className="text-sm font-semibold text-slate-200 mb-2">Calibration Profile</h3>
                        <input
                            type="file"
                            accept=".json,application/json"
                            onChange={(event) => onProfileFile(event.target.files?.[0] ?? null)}
                            className="block w-full text-xs text-slate-300"
                        />
                        {profileError && <p role="alert" className="mt-2 text-xs text-red-300">{profileError}</p>}
                        {calibrationProfile ? (
                            <div className="mt-2 rounded bg-slate-900/50 p-2 text-xs text-slate-300 space-y-1">
                                <div className="flex justify-between gap-2">
                                    <span>{calibrationProfile.name}</span>
                                    <button type="button" onClick={onClearProfile} className="text-red-300">Clear</button>
                                </div>
                                <div>Validation RMS: {Number.isFinite(calibrationProfile.validation.rms3d)
                                    ? `${calibrationProfile.validation.rms3d.toFixed(3)} m` : 'N/A'}</div>
                                <div>Re: {calibrationProfile.domain.reynolds.map((value) => Math.round(value)).join('–')}</div>
                                <div>S: {calibrationProfile.domain.spinParameter.map((value) => value.toFixed(3)).join('–')}</div>
                                {Number.isFinite(validationRms) && validationRms > 0.15 && (
                                    <div className="text-amber-300">Profile validation error is relatively high.</div>
                                )}
                                {clampedFraction > 0 && (
                                    <div className="text-amber-300">
                                        Outside calibrated domain for {(100 * clampedFraction).toFixed(1)}% of flight samples.
                                    </div>
                                )}
                            </div>
                        ) : (
                            <p className="mt-2 text-xs text-slate-500">Using uncalibrated FUEL baseline coefficients.</p>
                        )}
                    </section>

                    <section>
                        <label className="flex items-center justify-between gap-3 text-sm text-slate-200">
                            <span>Robust Analysis</span>
                            <input type="checkbox" checked={robustEnabled}
                                   onChange={(event) => onRobustEnabledChange(event.target.checked)}/>
                        </label>
                        {robustEnabled && (
                            <div className="mt-3 space-y-3">
                                <div className="grid grid-cols-2 gap-2">
                                    <NumberField label="Velocity σ" value={uncertaintyConfig.velocity.sigma}
                                                 onChange={(value) => setSigma('velocity', value)} min={0} step={0.05} unit="m/s"/>
                                    <NumberField label="Angle σ" value={uncertaintyConfig.angleDeg.sigma}
                                                 onChange={(value) => setSigma('angleDeg', value)} min={0} step={0.1} unit="°"/>
                                    <NumberField label="Spin σ" value={uncertaintyConfig.spinRPM.sigma}
                                                 onChange={(value) => setSigma('spinRPM', value)} min={0} step={25} unit="RPM"/>
                                    <NumberField label="Mass σ" value={uncertaintyConfig.mass.sigma}
                                                 onChange={(value) => setSigma('mass', value)} min={0} step={0.001} unit="kg"/>
                                    <NumberField label="Robot Forward σ" value={uncertaintyConfig.robotVelocity?.[0]?.sigma ?? 0}
                                                 onChange={(value) => setVectorSigma('robotVelocity', 0, value)}
                                                 min={0} step={0.05} unit="m/s"/>
                                    <NumberField label="Robot Lateral σ" value={uncertaintyConfig.robotVelocity?.[1]?.sigma ?? 0}
                                                 onChange={(value) => setVectorSigma('robotVelocity', 1, value)}
                                                 min={0} step={0.05} unit="m/s"/>
                                    <NumberField label="Drag multiplier σ" value={uncertaintyConfig.dragMultiplier.sigma}
                                                 onChange={(value) => setSigma('dragMultiplier', value)} min={0} step={0.01}/>
                                    <NumberField label="Lift multiplier σ" value={uncertaintyConfig.liftMultiplier.sigma}
                                                 onChange={(value) => setSigma('liftMultiplier', value)} min={0} step={0.01}/>
                                    <NumberField label="Seed" value={uncertaintySeed}
                                                 onChange={(value) => onUncertaintySeedChange(Math.trunc(value || 0))} step={1}/>
                                    <NumberField label="Final samples" value={robustSampleCount}
                                                 onChange={(value) => onRobustSampleCountChange(Math.max(1, Math.min(10000, Math.trunc(value || 1))))}
                                                 min={1} max={10000} step={64}/>
                                </div>
                                {robustResult && (
                                    <div className="rounded bg-slate-900/50 p-2 text-xs text-slate-300">
                                        <div>Clean-entry probability: {probability(robustResult.probabilities?.['clean-entry'])}</div>
                                        <div>10th-percentile clearance: {Number.isFinite(robustResult.clearance?.p10)
                                            ? `${(100 * robustResult.clearance.p10).toFixed(1)} cm` : 'N/A'}</div>
                                    </div>
                                )}
                            </div>
                        )}
                    </section>
                </div>
            )}
        </div>
    );
}
