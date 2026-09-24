import { Component, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './index.css';

/**
 * Last line of defence: a render error anywhere below would otherwise unmount the
 * whole tree and leave a blank page, which is indistinguishable from "the app is
 * broken and there is nothing I can do".
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
    this.handleReload = this.handleReload.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[ecosort] Unrecoverable render error:', error, info?.componentStack);
  }

  handleReload() {
    window.location.reload();
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="flex min-h-full items-center justify-center bg-slate-50 p-6 dark:bg-slate-950">
        <div className="card w-full max-w-lg p-6">
          <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            EcoSort hit an unexpected error
          </h1>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
            The interface stopped rendering. Reloading usually clears it — your saved
            history is stored in the backend database and is unaffected.
          </p>

          <pre className="mt-4 max-h-48 overflow-auto rounded-lg bg-slate-100 p-3 font-mono text-xs text-slate-800 dark:bg-slate-800 dark:text-slate-200">
            {String(error && error.message ? error.message : error)}
          </pre>

          <div className="mt-5 flex flex-wrap gap-2">
            <button type="button" className="btn btn-primary" onClick={this.handleReload}>
              Reload EcoSort
            </button>
          </div>

          <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
            If it keeps happening, check the browser console and the backend logs with{' '}
            <code className="font-mono">docker compose logs -f</code>.
          </p>
        </div>
      </div>
    );
  }
}

const container = document.getElementById('root');
if (!container) {
  throw new Error('EcoSort could not start: no #root element found in index.html.');
}

createRoot(container).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
