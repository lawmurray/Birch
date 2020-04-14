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
  return static_cast<bi::type::Real64>(x);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer64) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64>(x);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer32) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64>(x);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer16) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64>(x);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Integer8) -> Real64 {
  cpp{{
  return static_cast<bi::type::Real64>(x);
  }}
}

/**
 * Convert to Real64.
 */
function Real64(x:Boolean) -> Real64 {
  if x {
    return Real64(1.0);
  } else {
    return Real64(0.0);
  }
}

/**
 * Convert to Real64.
 */
function Real64(x:String) -> Real64 {
  cpp{{
  return ::strtod(x.c_str(), nullptr);
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
function Real64(x:Integer8?) -> Real64? {
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
  return std::abs(x);
  }}
}

/**
 * Power.
 */
function pow(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::pow(x, y);
  }}
}

/**
 * Modulus.
 */
function mod(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmod(x, y);
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::max(x, y);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::min(x, y);
  }}
}

/**
 * Is the value `inf`?
 */
function isinf(x:Real64) -> Boolean {
  cpp{{
  return std::isinf(x);
  }}
}

/**
 * Is the value `nan`?
 */
function isnan(x:Real64) -> Boolean {
  cpp{{
  return std::isnan(x);
  }}
}

/**
 * Is the value finite (i.e. not `inf` or `nan`)?
 */
function isfinite(x:Real64) -> Boolean {
  cpp{{
  return std::isfinite(x);
  }}
}
