import basic;

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

/**
 * Length of a string.
 */
function length(x:String) -> Integer {
  cpp{{
  return x_.length();
  }}
}
