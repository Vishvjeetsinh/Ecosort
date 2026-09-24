/** Terminal 404 for anything no router matched; uses the same envelope as errorHandler. */
export function notFoundHandler(req, res) {
  res.status(404).json({
    error: {
      code: 'not_found',
      message: `Cannot ${req.method} ${req.originalUrl}`,
      details: null,
    },
  });
}

export default notFoundHandler;
