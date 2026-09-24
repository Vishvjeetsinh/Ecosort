/**
 * Wraps an async route handler so a rejected promise reaches Express' error pipeline
 * instead of becoming an unhandled rejection (Express 4 does not await handlers).
 */
export const asyncHandler = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

export default asyncHandler;
