type Boolean;
type Real64;
type Real32;
type Integer64;
type Integer32;
type String;

type Real = Real64;
type Integer = Integer64;

/**
 * Convert String to Boolean.
 */
function Boolean(s:String) -> Boolean {
  cpp{{
  return ::atoi(s_.c_str());
  }}
}

/**
 * Convert String to Real64.
 */
function Real64(s:String) -> Real64 {
  cpp{{
  return ::strtod(s_.c_str(), nullptr);
  }}
}

/**
 * Convert String to Real32.
 */
function Real32(s:String) -> Real32 {
  cpp{{
  return ::strtof(s_.c_str(), nullptr);
  }}
}

/**
 * Convert String to Integer64.
 */
function Integer64(s:String) -> Integer64 {
  cpp{{
  return ::atol(s_.c_str());
  }}
}

/**
 * Convert String to Integer32.
 */
function Integer32(s:String) -> Integer32 {
  cpp{{
  return ::atoi(s_.c_str());
  }}
}

/**
 * Convert String to Real.
 */
function Real(s:String) -> Real {
  cpp{{
  return ::strtod(s_.c_str(), nullptr);
  }}
}

/**
 * Convert Integer to Real.
 */
function Real(x:Integer) -> Real {
  cpp{{
  return static_cast<bi::type::Real_>(x_);
  }}
}

/**
 * Convert Reak to Integer.
 */
function Integer(x:Real) -> Integer {
  cpp{{
  return static_cast<bi::type::Integer_>(x_);
  }}
}

/**
 * Convert String to Integer.
 */
function Integer(s:String) -> Integer {
  cpp{{
  return ::atol(s_.c_str());
  }}
}

/**
 * Convert String to String (identity function).
 */
function String(s:String) -> String {
  return s;
}
