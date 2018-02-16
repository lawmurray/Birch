/**
 * Default floating point.
 */
type Real = Real64;

/**
 * Convert other basic types to Real. This is overloaded for Real64,
 * Real32, Integer64, Integer32 and String.
 */
function Real(x:Real64) -> Real {
  return Real64(x);
}
function Real(x:Real32) -> Real {
  return Real64(x);
}
function Real(x:Integer64) -> Real {
  return Real64(x);
}
function Real(x:Integer32) -> Real {
  return Real64(x);
}
function Real(x:String) -> Real {
  return Real64(x);
}
