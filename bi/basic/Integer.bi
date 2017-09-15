/**
 * An integer value of default type.
 */
type Integer = Integer64;

/**
 * Convert other basic types to Integer. This is overloaded for Real64,
 * Real32, Integer64, Integer32 and String.
 */
function Integer(x:Integer64) -> Integer {
  return Integer64(x);
}
function Integer(x:Real64) -> Integer {
  return Integer64(x);
}
function Integer(x:Real32) -> Integer {
  return Integer64(x);
}
function Integer(x:Integer32) -> Integer {
  return Integer64(x);
}
function Integer(x:String) -> Integer {
  return Integer64(x);
}
