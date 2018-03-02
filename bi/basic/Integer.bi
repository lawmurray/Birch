/**
 * Default integer.
 */
type Integer = Integer64;

/**
 * Convert to Integer.
 */
function Integer(x:Real64) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Real32) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer64) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer32) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer16) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer8) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:String) -> Integer {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Real64?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Real32?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer64?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer32?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer16?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:Integer8?) -> Integer? {
  return Integer64(x);
}

/**
 * Convert to Integer.
 */
function Integer(x:String?) -> Integer? {
  return Integer64(x);
}
