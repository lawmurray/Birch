/**
 * A 32-bit (single precision) floating point value.
 */
type Real32;

/**
 * Convert other basic types to Real32. This is overloaded for Real64, Real32,
 * Integer64, Integer32 and String.
 */
function Real32(x:Real32) -> Real32 {
  return x;
}
function Real32(x:Real64) -> Real32 {
  cpp{{
  return static_cast<bi::Real32_>(x_);
  }}
}
function Real32(x:Integer64) -> Real32 {
  cpp{{
  return static_cast<bi::Real32_>(x_);
  }}
}
function Real32(x:Integer32) -> Real32 {
  cpp{{
  return static_cast<bi::Real32_>(x_);
  }}
}
function Real32(s:String) -> Real32 {
  cpp{{
  return ::strtof(s_.c_str(), nullptr);
  }}
}

/*
 * Operators
 */
operator x:Real32 + y:Real32 -> Real32;
operator x:Real32 - y:Real32 -> Real32;
operator x:Real32 * y:Real32 -> Real32;
operator x:Real32 / y:Real32 -> Real32;
operator +x:Real32 -> Real32;
operator -x:Real32 -> Real32;
operator x:Real32 > y:Real32 -> Boolean;
operator x:Real32 < y:Real32 -> Boolean;
operator x:Real32 >= y:Real32 -> Boolean;
operator x:Real32 <= y:Real32 -> Boolean;
operator x:Real32 == y:Real32 -> Boolean;
operator x:Real32 != y:Real32 -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Real32) -> Real32 {
  cpp {{
  return std::abs(x_);
  }}
}

/**
 * Modulus.
 */
function mod(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fmodf(x_, y_);
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::max(x_, y_);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::min(x_, y_);
  }}
}

/**
 * Does this have the value NaN?
 */
function isnan(x:Real32) -> Boolean {
  return x != x;
}
