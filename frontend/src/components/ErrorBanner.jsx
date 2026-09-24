/**
 * Renders an `ApiError` (or any Error) with a retry affordance.
 *
 * @param {{title?:string, error:Error|null, onRetry?:Function, onDismiss?:Function}} props
 */
export function ErrorBanner({ title = 'Something went wrong', error, onRetry, onDismiss }) {
  if (!error) return null;

  const message =
    (error && error.message) || 'An unexpected error occurred with no further detail.';
  // status 0 means the request never reached the server at all.
  const offline = error && error.status === 0;

  return (
    <div
      role="alert"
      className="animate-fade-in rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-100"
    >
      <div className="flex items-start gap-3">
        <svg
          viewBox="0 0 24 24"
          className="mt-0.5 h-5 w-5 shrink-0 text-red-600 dark:text-red-400"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M12 9v4m0 4h.01M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20.2h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <div className="min-w-0 flex-1">
          <p className="font-semibold">{title}</p>
          <p className="mt-0.5 break-words">{message}</p>

          {offline ? (
            <p className="mt-2 text-xs text-red-800 dark:text-red-200">
              The EcoSort API did not answer. Check that the backend container is up:{' '}
              <code className="rounded bg-red-100 px-1 py-0.5 font-mono dark:bg-red-900/50">
                docker compose ps
              </code>{' '}
              and{' '}
              <code className="rounded bg-red-100 px-1 py-0.5 font-mono dark:bg-red-900/50">
                docker compose logs backend
              </code>
              .
            </p>
          ) : null}

          {error && error.code ? (
            <p className="mt-2 font-mono text-[11px] uppercase tracking-wide opacity-70">
              {error.code}
              {error.status ? ` · HTTP ${error.status}` : ''}
            </p>
          ) : null}

          {onRetry || onDismiss ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {onRetry ? (
                <button type="button" className="btn btn-ghost" onClick={onRetry}>
                  Try again
                </button>
              ) : null}
              {onDismiss ? (
                <button type="button" className="btn btn-ghost" onClick={onDismiss}>
                  Dismiss
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default ErrorBanner;
