import { useCallback, useEffect, useRef } from 'react';

/**
 * WAI-ARIA tablist with roving tabindex: only the selected tab is in the tab order,
 * and arrow/Home/End move selection between tabs.
 *
 * Panels are rendered by the parent and must carry `role="tabpanel"` plus
 * `id={panelId(item.id)}` / `aria-labelledby={tabId(item.id)}`.
 *
 * @param {{value:string, onChange:Function,
 *          items:Array<{id:string,label:string,icon?:React.ReactNode,badge?:React.ReactNode}>,
 *          label?:string}} props
 */
export function Tabs({ value, onChange, items = [], label = 'Sections' }) {
  const refs = useRef(new Map());
  // Focus must only follow a keyboard-driven selection change, never a render.
  const pendingFocus = useRef(null);

  useEffect(() => {
    const id = pendingFocus.current;
    if (!id) return;
    pendingFocus.current = null;
    const node = refs.current.get(id);
    if (node) node.focus();
  }, [value]);

  const select = useCallback(
    (id, withFocus) => {
      // Always reset first: a stale pending id would otherwise steal focus on the
      // next selection change (e.g. keyboard move, then a click elsewhere).
      pendingFocus.current = withFocus ? id : null;

      if (id === value || !onChange) {
        if (withFocus) {
          const node = refs.current.get(id);
          if (node) node.focus();
        }
        pendingFocus.current = null;
        return;
      }
      onChange(id);
    },
    [onChange, value],
  );

  const onKeyDown = useCallback(
    (event) => {
      const index = items.findIndex((item) => item.id === value);
      if (index < 0 || items.length === 0) return;

      let nextIndex = null;
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          nextIndex = (index + 1) % items.length;
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          nextIndex = (index - 1 + items.length) % items.length;
          break;
        case 'Home':
          nextIndex = 0;
          break;
        case 'End':
          nextIndex = items.length - 1;
          break;
        default:
          return;
      }

      event.preventDefault();
      select(items[nextIndex].id, true);
    },
    [items, select, value],
  );

  return (
    <div
      role="tablist"
      aria-label={label}
      onKeyDown={onKeyDown}
      className="flex w-full gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1 dark:border-slate-800 dark:bg-slate-900"
    >
      {items.map((item) => {
        const selected = item.id === value;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={tabId(item.id)}
            aria-selected={selected}
            aria-controls={panelId(item.id)}
            tabIndex={selected ? 0 : -1}
            ref={(node) => {
              if (node) refs.current.set(item.id, node);
              else refs.current.delete(item.id);
            }}
            onClick={() => select(item.id, false)}
            className={[
              'flex flex-1 shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium transition-colors',
              selected
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800',
            ].join(' ')}
          >
            {item.icon ? (
              <span aria-hidden="true" className="flex items-center">
                {item.icon}
              </span>
            ) : null}
            <span>{item.label}</span>
            {item.badge !== undefined && item.badge !== null && item.badge !== '' ? (
              <span
                className={[
                  'rounded-full px-1.5 py-0.5 text-[11px] font-semibold tabular-nums',
                  selected
                    ? 'bg-white/25 text-white'
                    : 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200',
                ].join(' ')}
              >
                {item.badge}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

export function tabId(id) {
  return `ecosort-tab-${id}`;
}

export function panelId(id) {
  return `ecosort-panel-${id}`;
}

export default Tabs;
