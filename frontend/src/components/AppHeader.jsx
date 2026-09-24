import ModelSelect from './ModelSelect.jsx';
import RegionSelect from './RegionSelect.jsx';

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8 shrink-0" aria-hidden="true">
      <rect width="32" height="32" rx="7" className="fill-brand-600" />
      <path d="M16 4.6l4.7 8.1h-3.3v4.9h-2.8v-4.9h-3.3L16 4.6z" fill="#ffffff" />
      <path
        d="M6.4 22.9l4.1-7.1 2.4 1.4-2.7 4.7 4.7 2.7-1.4 2.4-7.1-4.1z"
        fill="#ffffff"
        fillOpacity="0.85"
      />
      <path
        d="M25.6 22.9l-7.1 4.1-1.4-2.4 4.7-2.7-2.7-4.7 2.4-1.4 4.1 7.1z"
        fill="#ffffff"
        fillOpacity="0.7"
      />
    </svg>
  );
}

/**
 * The header carries both of the app's global choices — which region's rules apply and
 * which model classifies — plus the theme and settings affordances.
 *
 * ModelStatusBadge used to sit here; its dot, name and progress are now folded into
 * ModelSelect so the loaded model is named exactly once. The badge component is
 * unchanged in behaviour for anyone else who wants the standalone chip.
 *
 * @param {{modelStatus:object|null, engineKind:string, engineStatus:string,
 *          engineProgress?:object|null, engineDisplayName?:string,
 *          models?:Array<object>, modelId?:string, onModelChange?:Function,
 *          requestedModelId?:string,
 *          regions:Array, regionId:string, onRegionChange:Function,
 *          onOpenSettings:Function, theme:'light'|'dark'|'system', onToggleTheme:Function}} props
 */
export function AppHeader({
  modelStatus,
  engineKind,
  engineStatus,
  engineProgress,
  engineDisplayName = '',
  models = [],
  modelId = '',
  onModelChange,
  requestedModelId = '',
  activeModelId = '',
  regions = [],
  regionId,
  onRegionChange,
  onOpenSettings,
  theme = 'light',
  onToggleTheme,
}) {
  // theme is 'light' | 'dark' | 'system'. Under 'system' the button's icon and label have
  // to describe what is actually on screen, which means asking the OS — otherwise a dark
  // desktop shows a "switch to dark" control that switches to light.
  const dark =
    theme === 'dark' ||
    (theme === 'system' &&
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
  const activeKind = engineKind || modelStatus?.active || 'none';

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
      <div className="mx-auto flex w-full max-w-shell flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2.5 sm:px-5">
        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <Logo />
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold leading-tight sm:text-lg">
              EcoSort
            </h1>
            <p className="hidden truncate text-xs text-slate-500 dark:text-slate-400 sm:block">
              Offline waste sorting assistant
            </p>
          </div>
        </div>

        {/* Below sm the two pickers stack full width so neither is squeezed to a stub;
            from sm they sit side by side and the row wraps as a whole. */}
        <div className="order-3 flex w-full min-w-0 flex-col gap-2 sm:order-none sm:w-auto sm:flex-row sm:items-start sm:gap-3">
          <div className="min-w-0 sm:w-48">
            <RegionSelect
              regions={regions}
              value={regionId}
              onChange={onRegionChange}
              compact
              label="Recycling region"
            />
          </div>

          <div className="min-w-0 sm:w-72">
            <ModelSelect
              models={models}
              value={modelId}
              onChange={onModelChange}
              status={engineStatus}
              kind={activeKind}
              displayName={engineDisplayName}
              progress={engineProgress}
              requestedModelId={requestedModelId}
              activeModelId={activeModelId}
            />
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            className="btn-icon"
            onClick={onToggleTheme}
            aria-pressed={dark}
            title={dark ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {dark ? (
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" aria-hidden="true">
                <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.8" />
                <path
                  d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.1 5.1l1.4 1.4M17.5 17.5l1.4 1.4M18.9 5.1l-1.4 1.4M6.5 17.5l-1.4 1.4"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" aria-hidden="true">
                <path
                  d="M20.5 14.3A8.5 8.5 0 0 1 9.7 3.5a8.5 8.5 0 1 0 10.8 10.8Z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="btn-icon"
            onClick={onOpenSettings}
            title="Open settings"
            aria-label="Open settings"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.8" />
              <path
                d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}

export default AppHeader;
