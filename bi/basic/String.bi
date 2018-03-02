/**
 * Character string.
 */
type String;

/**
 * Convert to string.
 */
function String(x:Boolean) -> String {
  if (x) {
    return "true";
  } else {
    return "false";
  }
}

/**
 * Convert to string.
 */
function String(x:Real64) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:Real32) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer64) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer32) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer16) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer8) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/**
 * Convert to string.
 */
function String(x:String) -> String {
  return x;
}

/**
 * Convert to string.
 */
function String(x:Boolean?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Real64?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Real32?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Integer64?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Integer32?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Integer16?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:Integer8?) -> String? {
  if (x?) {
    return String(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to string.
 */
function String(x:String?) -> String? {
  return x;
}

/**
 * Lexical comparison.
 */
operator (x:String > y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) > 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String < y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) < 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String >= y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) >= 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String <= y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) <= 0;
  }}
}

/**
 * Equality comparison.
 */
operator (x:String == y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) == 0;
  }}
}

/**
 * Equality comparison.
 */
operator (x:String != y:String) -> Boolean {
  cpp{{
  return x_.compare(y_) != 0;
  }}
}

/**
 * String concatenation.
 */
operator (x:String + y:String) -> String;

/**
 * String concatenation.
 */
operator (x:String + y:Boolean) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Real64) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Real32) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Integer64) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Integer32) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:Boolean + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Real64 + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Real32 + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Integer64 + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Integer32 + y:String) -> String {
  return String(x) + y;
}

/**
 * Length of a string.
 */
function length(x:String) -> Integer {
  cpp{{
  return x_.length();
  }}
}
