/**
 * 64-bit integer.
 */
type Integer64 < Real64;

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer64) -> Integer64 {
  return x;
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Real64) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64_>(x_);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Real32) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64_>(x_);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer32) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64_>(x_);
  }}
}

/**
 * Convert to Integer64.
 */
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

/**
 * Round an integer up to the next power of two. Zero and negative integers
 * return zero.
 */
function pow2(x:Integer64) -> Integer64 {
  if (x < 0) {
    return Integer64(0);
  } else {
    y:Integer64 <- x/2;
    z:Integer64 <- 1;
    while (y > 0) {
      y <- y/2;
      z <- z*2;
    }
    return z;
  }
}
