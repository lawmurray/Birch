/**
 * A 64-bit (double precision) floating point value.
 */
type Real64;

/**
 * Convert other basic types to Real64. This is overloaded for Real64, Real32,
 * Integer64, Integer32 and String.
 */
function Real64(x:Real64) -> Real64 {
  return x;
}
function Real64(x:Real32) -> Real64 {
  cpp{{
  return static_cast<bi::Real64_>(x_);
  }}
}
function Real64(x:Integer64) -> Real64 {
  cpp{{
  return static_cast<bi::Real64_>(x_);
  }}
}
function Real64(x:Integer32) -> Real64 {
  cpp{{
  return static_cast<bi::Real64_>(x_);
  }}
}
function Real64(s:String) -> Real64 {
  cpp{{
  return ::strtod(s_.c_str(), nullptr);
  }}
}

/*
 * Operators
 */
operator x:Real64 + y:Real64 -> Real64;
operator x:Real64 - y:Real64 -> Real64;
operator x:Real64 * y:Real64 -> Real64;
operator x:Real64 / y:Real64 -> Real64;
operator +x:Real64 -> Real64;
operator -x:Real64 -> Real64;
operator x:Real64 > y:Real64 -> Boolean;
operator x:Real64 < y:Real64 -> Boolean;
operator x:Real64 >= y:Real64 -> Boolean;
operator x:Real64 <= y:Real64 -> Boolean;
operator x:Real64 == y:Real64 -> Boolean;
operator x:Real64 != y:Real64 -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Real64) -> Real64 {
  cpp {{
  return std::abs(x_);
  }}
}

/**
 * Modulus.
 */
function mod(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmod(x_, y_);
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::max(x_, y_);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::min(x_, y_);
  }}
}

/**
 * Does this have the value NaN?
 */
function isnan(x:Real64) -> Boolean {
  return x != x;
}
