const TONE_STYLES = {
  success:
    'border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/70 dark:text-emerald-100',
  error:
    'border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/70 dark:text-red-100',
  info: 'border-slate-300 bg-white text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100',
};

const TONE_ICONS = {
  success: 'M20 6 9 17l-5-5',
  error: 'M18 6 6 18M6 6l12 12',
  info: 'M12 16v-5m0-4h.01',
};

/**
 * @param {{toasts: Array<{id:number,title:string,message:string,tone:string}>,
 *          onDismiss: Function}} props
 */
export function Toaster({ toasts = [], onDismiss }) {
  return (
    <div
      aria-live="polite"
      aria-relevant="additions text"
      className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex flex-col items-end gap-2 p-3 sm:inset-x-auto sm:right-0"
    >
      {toasts.map((toast) => {
        const tone = TONE_STYLES[toast.tone] ? toast.tone : 'info';
        return (
          <div
            key={toast.id}
            className={`pointer-events-auto flex w-full max-w-sm animate-fade-in items-start gap-3 rounded-xl border p-3 shadow-lg ${TONE_STYLES[tone]}`}
          >
            <svg viewBox="0 0 24 24" className="mt-0.5 h-5 w-5 shrink-0" fill="none" aria-hidden="true">
              <path
                d={TONE_ICONS[tone]}
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>

            <div className="min-w-0 flex-1">
              {toast.title ? <p className="text-sm font-semibold">{toast.title}</p> : null}
              {toast.message ? (
                <p className="mt-0.5 break-words text-sm opacity-90">{toast.message}</p>
              ) : null}
            </div>

            <button
              type="button"
              onClick={() => onDismiss && onDismiss(toast.id)}
              className="-m-1 rounded p-1 opacity-60 transition-opacity hover:opacity-100"
              aria-label={`Dismiss notification${toast.title ? `: ${toast.title}` : ''}`}
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" aria-hidden="true">
                <path
                  d="M18 6 6 18M6 6l12 12"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default Toaster;
