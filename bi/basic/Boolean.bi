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
 * Convert to Boolean.
 */
function Boolean(x:Boolean?) -> Boolean? {
  return x;
}

/**
 * Convert to Boolean.
 */
function Boolean(x:String?) -> Boolean? {
  if (x?) {
    return Boolean(x!);
  } else {
    return nil;
  }
}

/**
 * Logical *and*.
 */
operator (x:Boolean && y:Boolean) -> Boolean;

/**
 * Logical *and*.
 */
operator (x:Boolean * y:Boolean) -> Boolean;

/**
 * Logical *or*.
 */
operator (x:Boolean || y:Boolean) -> Boolean;

/**
 * Logical *or*.
 */
operator (x:Boolean + y:Boolean) -> Boolean;

/**
 * Logical *not*.
 */
operator (!x:Boolean) -> Boolean;

/**
 * Maximum of two values (logical *or*).
 */
function max(x:Boolean, y:Boolean) -> Boolean {
  return x || y;
}

/**
 * Minimum of two values (logical *and*);
 */
function min(x:Boolean, y:Boolean) -> Boolean {
  return x && y;
}
