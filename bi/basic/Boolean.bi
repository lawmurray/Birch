/**
 * Boolean.
 */
type Boolean;

/**
 * Convert to Boolean.
 */
function Boolean(x:Boolean) -> Boolean {
  return x;
}

/**
 * Convert to Boolean.
 */
function Boolean(x:String) -> Boolean {
  return x == "true";
}

/**
 * Logical *and*.
 */
operator (x:Boolean && y:Boolean) -> Boolean;

/**
 * Logical *or*.
 */
operator (x:Boolean || y:Boolean) -> Boolean;

/**
 * Logical *not*.
 */
operator (!x:Boolean) -> Boolean;
