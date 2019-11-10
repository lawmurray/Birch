/**
 * Character string.
 */
type String;

/**
 * Convert to string.
 */
function String(x:Boolean) -> String {
  if x {
    return "true";
  } else {
    return "false";
  }
}

/**
 * Convert double-precision floating point number to string.
 */
function String(x:Real64) -> String {
  if x == 0.0 {
    return "0.0";
  } else {
    cpp{{
    std::stringstream buf;
    buf << std::scientific << std::setprecision(14) << x;
  
    /* remove trailing zeros */
    auto str = buf.str();
    auto i = str.find('.');
    if (i != std::string::npos) {
      auto j = str.find('e', i);
      if (j != std::string::npos) {
        auto k = str.find_last_not_of('0', j - 1);
        if (k != std::string::npos) {
          auto split = std::max(i + 1, k);
          return str.substr(0, split + 1) + str.substr(j, str.length() - i);
        }
      }
    }
    return str;
    }}
  }
}

/**
 * Convert single-precision floating point number to string.
 */
function String(x:Real32) -> String {
  result:String;
  if x == 0.0 {
    return "0.0";
  } else {
    cpp{{
    std::stringstream buf;
    buf << std::scientific << std::setprecision(6) << x;
  
    /* remove trailing zeros */
    auto str = buf.str();
    auto i = str.find('.');
    if (i != std::string::npos) {
      auto j = str.find('e', i);
      if (j != std::string::npos) {
        auto k = str.find_last_not_of('0', j - 1);
        if (k != std::string::npos) {
          auto split = std::max(i + 1, k);
          return str.substr(0, split + 1) + str.substr(j, str.length() - i);
        }
      }
    }
    return str;
    }}
  }
}

/**
 * Convert to string.
 */
function String(x:Integer64) -> String {
  cpp{{
  return std::to_string(x);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer32) -> String {
  cpp{{
  return std::to_string(x);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer16) -> String {
  cpp{{
  return std::to_string(x);
  }}
}

/**
 * Convert to string.
 */
function String(x:Integer8) -> String {
  cpp{{
  return std::to_string(x);
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
  return x.compare(y) > 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String < y:String) -> Boolean {
  cpp{{
  return x.compare(y) < 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String >= y:String) -> Boolean {
  cpp{{
  return x.compare(y) >= 0;
  }}
}

/**
 * Lexical comparison.
 */
operator (x:String <= y:String) -> Boolean {
  cpp{{
  return x.compare(y) <= 0;
  }}
}

/**
 * Equality comparison.
 */
operator (x:String == y:String) -> Boolean {
  cpp{{
  return x.compare(y) == 0;
  }}
}

/**
 * Equality comparison.
 */
operator (x:String != y:String) -> Boolean {
  cpp{{
  return x.compare(y) != 0;
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
operator (x:String + y:Real) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Integer) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Boolean[_]) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Real[_]) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Integer[_]) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Boolean[_,_]) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Real[_,_]) -> String {
  return x + String(y);
}

/**
 * String concatenation.
 */
operator (x:String + y:Integer[_,_]) -> String {
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
operator (x:Real + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Integer + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Boolean[_] + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Real[_] + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Integer[_] + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Boolean[_,_] + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Real[_,_] + y:String) -> String {
  return String(x) + y;
}

/**
 * String concatenation.
 */
operator (x:Integer[_,_] + y:String) -> String {
  return String(x) + y;
}

/**
 * Length of a string.
 */
function length(x:String) -> Integer {
  cpp{{
  return x.length();
  }}
}

/**
 * Length of an array of strings.
 */
function length(x:String[_]) -> Integer {
  cpp{{
  return x.rows();
  }}
}