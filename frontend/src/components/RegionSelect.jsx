import { useId } from 'react';

/**
 * @param {{regions: Array<{id:string,name:string,country:string}>, value:string,
 *          onChange:Function, disabled?:boolean, label?:string,
 *          compact?:boolean, className?:string}} props
 */
export function RegionSelect({
  regions = [],
  value = '',
  onChange,
  disabled = false,
  label = 'Recycling region',
  compact = false,
  className = '',
}) {
  const id = useId();
  const empty = regions.length === 0;

  return (
    <div className={`flex min-w-0 flex-col gap-1 ${className}`.trim()}>
      <label htmlFor={id} className={compact ? 'sr-only' : 'label'}>
        {label}
      </label>
      <select
        id={id}
        className="input py-1.5"
        value={value || ''}
        disabled={disabled || empty}
        onChange={(event) => onChange && onChange(event.target.value)}
        aria-label={compact ? label : undefined}
      >
        {empty ? (
          <option value="">No regions available</option>
        ) : (
          <>
            {value ? null : <option value="">Select a region…</option>}
            {regions.map((region) => (
              <option key={region.id} value={region.id}>
                {region.country ? `${region.name} — ${region.country}` : region.name}
              </option>
            ))}
          </>
        )}
      </select>
    </div>
  );
}

export default RegionSelect;
