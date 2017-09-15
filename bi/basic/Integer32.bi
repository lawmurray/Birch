/**
 * A 32-bit integer.
 */
type Integer32;

/**
 * Convert other basic types to Integer32. This is overloaded for Real64,
 * Real32, Integer64, Integer32 and String.
 */
function Integer32(x:Integer32) -> Integer32 {
  return x;
}
function Integer32(x:Real64) -> Integer32 {
  cpp{{
  return static_cast<bi::Integer32_>(x_);
  }}
}
function Integer32(x:Real32) -> Integer32 {
  cpp{{
  return static_cast<bi::Integer32_>(x_);
  }}
}
function Integer32(x:Integer64) -> Integer32 {
  cpp{{
  return static_cast<bi::Integer32_>(x_);
  }}
}
function Integer32(s:String) -> Integer32 {
  cpp{{
  return ::atoi(s_.c_str());
  }}
}

/*
 * Operators
 */
operator x:Integer32 + y:Integer32 -> Integer32;
operator x:Integer32 - y:Integer32 -> Integer32;
operator x:Integer32 * y:Integer32 -> Integer32;
operator x:Integer32 / y:Integer32 -> Integer32;
operator +x:Integer32 -> Integer32;
operator -x:Integer32 -> Integer32;
operator x:Integer32 > y:Integer32 -> Boolean;
operator x:Integer32 < y:Integer32 -> Boolean;
operator x:Integer32 >= y:Integer32 -> Boolean;
operator x:Integer32 <= y:Integer32 -> Boolean;
operator x:Integer32 == y:Integer32 -> Boolean;
operator x:Integer32 != y:Integer32 -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Integer32) -> Integer32 {
  cpp {{
  return std::abs(x_);
  }}
}

/**
 * Modulus.
 */
function mod(x:Integer32, y:Integer32) -> Integer32 {
  cpp {{
  return x_ % y_;
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Integer32, y:Integer32) -> Integer32 {
  cpp {{
  return std::max(x_, y_);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Integer32, y:Integer32) -> Integer32 {
  cpp {{
  return std::min(x_, y_);
  }}
}
