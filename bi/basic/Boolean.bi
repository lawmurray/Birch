/**
 * Boolean.
 */
type Boolean;

/**
 * Convert other basic types to Boolean. This is overloaded for Boolean and
 * String.
 */
function Boolean(x:Boolean) -> Boolean {
  return x;
}
function Boolean(x:String) -> Boolean {
  return x == "true";
}

/*
 * Operators
 */
operator (x:Boolean && y:Boolean) -> Boolean;
operator (x:Boolean || y:Boolean) -> Boolean;
operator (!x:Boolean) -> Boolean;
