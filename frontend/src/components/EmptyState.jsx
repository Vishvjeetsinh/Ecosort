/**
 * @param {{icon?:React.ReactNode, title:string, message?:React.ReactNode,
 *          action?:React.ReactNode, className?:string}} props
 */
export function EmptyState({ icon, title, message, action, className = '' }) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-300 px-5 py-10 text-center dark:border-slate-700 ${className}`.trim()}
    >
      {icon ? (
        <div className="text-3xl text-slate-400 dark:text-slate-500" aria-hidden="true">
          {icon}
        </div>
      ) : null}
      <h3 className="text-base text-slate-800 dark:text-slate-100">{title}</h3>
      {message ? (
        <p className="max-w-prose text-sm text-slate-600 dark:text-slate-400">{message}</p>
      ) : null}
      {action ? <div className="pt-1">{action}</div> : null}
    </div>
  );
}

export default EmptyState;
