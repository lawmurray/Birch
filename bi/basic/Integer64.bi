/**
 * A 64-bit integer.
 */
type Integer64 < Real64;

/**
 * Convert other basic types to Integer64. This is overloaded for Real64,
 * Real32, Integer64, Integer32 and String.
 */
function Integer64(x:Integer64) -> Integer64 {
  return x;
}
function Integer64(x:Real64) -> Integer64 {
  cpp{{
  return static_cast<bi::Integer64_>(x_);
  }}
}
function Integer64(x:Real32) -> Integer64 {
  cpp{{
  return static_cast<bi::Integer64_>(x_);
  }}
}
function Integer64(x:Integer32) -> Integer64 {
  cpp{{
  return static_cast<bi::Integer64_>(x_);
  }}
}
function Integer64(s:String) -> Integer64 {
  cpp{{
  return ::atol(s_.c_str());
  }}
}

/*
 * Operators
 */
operator (x:Integer64 + y:Integer64) -> Integer64;
operator (x:Integer64 - y:Integer64) -> Integer64;
operator (x:Integer64 * y:Integer64) -> Integer64;
operator (x:Integer64 / y:Integer64) -> Integer64;
operator (+x:Integer64) -> Integer64;
operator (-x:Integer64) -> Integer64;
operator (x:Integer64 > y:Integer64) -> Boolean;
operator (x:Integer64 < y:Integer64) -> Boolean;
operator (x:Integer64 >= y:Integer64) -> Boolean;
operator (x:Integer64 <= y:Integer64) -> Boolean;
operator (x:Integer64 == y:Integer64) -> Boolean;
operator (x:Integer64 != y:Integer64) -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Integer64) -> Integer64 {
  cpp {{
  return std::abs(x_);
  }}
}

/**
 * Modulus.
 */
function mod(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return x_ % y_;
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::max(x_, y_);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::min(x_, y_);
  }}
}
