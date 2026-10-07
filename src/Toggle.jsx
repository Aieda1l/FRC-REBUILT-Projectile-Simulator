export default function Toggle({label, checked, onChange}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer mb-2 select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="peer sr-only"
      />
      <span
        aria-hidden="true"
        className="relative w-10 h-5 rounded-full bg-slate-600 transition-colors peer-checked:bg-indigo-500 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-cyan-400"
      >
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${checked ? 'translate-x-5' : 'translate-x-0.5'}`} />
      </span>
      <span className="text-sm text-slate-300">{label}</span>
    </label>
  );
}
