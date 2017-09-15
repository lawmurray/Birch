/**
 * A string value.
 */
type String;

/**
 * Convert other basic types to String. This is overloaded for Bolean, Real64,
 * String, Integer64, Integer32 and String.
 */
function String(x:String) -> String {
  return x;
}
function String(x:Boolean) -> String {
  if (x) {
    return "true";
  } else {
    return "false";
  }
}
function String(x:Real64) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}
function String(x:Real32) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}
function String(x:Integer64) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}
function String(x:Integer32) -> String {
  cpp{{
  return std::to_string(x_);
  }}
}

/*
 * Alphabetical string comparisons.
 */
operator x:String > y:String -> Boolean {
  cpp{{
  return x_.compare(y_) > 0;
  }}
}
operator x:String < y:String -> Boolean {
  cpp{{
  return x_.compare(y_) < 0;
  }}
}
operator x:String >= y:String -> Boolean {
  cpp{{
  return x_.compare(y_) >= 0;
  }}
}
operator x:String <= y:String -> Boolean {
  cpp{{
  return x_.compare(y_) <= 0;
  }}
}
operator x:String == y:String -> Boolean {
  cpp{{
  return x_.compare(y_) == 0;
  }}
}
operator x:String != y:String -> Boolean {
  cpp{{
  return x_.compare(y_) != 0;
  }}
}

/**
 * String concatenation.
 */
operator x:String + y:String -> String;
operator x:String + y:Boolean -> String {
  return x + String(y);
}
operator x:String + y:Real64 -> String {
  return x + String(y);
}
operator x:String + y:Real32 -> String {
  return x + String(y);
}
operator x:String + y:Integer64 -> String {
  return x + String(y);
}
operator x:String + y:Integer32 -> String {
  return x + String(y);
}
operator x:Boolean + y:String -> String {
  return String(x) + y;
}
operator x:Real64 + y:String -> String {
  return String(x) + y;
}
operator x:Real32 + y:String -> String {
  return String(x) + y;
}
operator x:Integer64 + y:String -> String {
  return String(x) + y;
}
operator x:Integer32 + y:String -> String {
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
