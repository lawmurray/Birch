/**
 * 64-bit (double precision) floating point.
 */
type Real64;

/**
 * Convert to Real64.
 */
function Real64(x:Real64) -> Real64 {
  return x;
}

/**
 * Convert to Real64.
 */
function Real64(x:Real32) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64_>(x_);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer64) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64_>(x_);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer32) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64_>(x_);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer16) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64_>(x_);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:String) -> Real64 {
  cpp{{
  return ::strtod(x_.c_str(), nullptr);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Real64?) -> Real64? {
  return x;
}

/**
 * Convert to Real64.
 */
function Real64(x:Real32?) -> Real64? {
  if (x?) {
    return Real64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer64?) -> Real64? {
  if (x?) {
    return Real64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer32?) -> Real64? {
  if (x?) {
    return Real64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer16?) -> Real64? {
  if (x?) {
    return Real64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real64.
 */
function Real64(x:String?) -> Real64? {
  if (x?) {
    return Real64(x!);
  } else {
    return nil;
  }
}

/*
 * Operators
 */
operator (x:Real64 + y:Real64) -> Real64;
operator (x:Real64 - y:Real64) -> Real64;
operator (x:Real64 * y:Real64) -> Real64;
operator (x:Real64 / y:Real64) -> Real64;
operator (+x:Real64) -> Real64;
operator (-x:Real64) -> Real64;
operator (x:Real64 > y:Real64) -> Boolean;
operator (x:Real64 < y:Real64) -> Boolean;
operator (x:Real64 >= y:Real64) -> Boolean;
operator (x:Real64 <= y:Real64) -> Boolean;
operator (x:Real64 == y:Real64) -> Boolean;
operator (x:Real64 != y:Real64) -> Boolean;

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
