/**
 * 32-bit (single precision) floating point.
 */
type Real32 < Real64;

/**
 * Convert to Real32.
 */
function Real32(x:Real64) -> Real32 {
  cpp{{
  return static_cast<bi::type::Real32>(x);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:Real32) -> Real32 {
  return x;
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer64) -> Real32 {
  cpp{{
  return static_cast<bi::type::Real32>(x);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer32) -> Real32 {
  cpp{{
  return static_cast<bi::type::Real32>(x);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer16) -> Real32 {
  cpp{{
  return static_cast<bi::type::Real32>(x);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer8) -> Real32 {
  cpp{{
  return static_cast<bi::type::Real32>(x);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:String) -> Real32 {
  cpp{{
  return ::strtof(x.c_str(), nullptr);
  }}
}

/**
 * Convert to Real32.
 */
function Real32(x:Real64?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real32.
 */
function Real32(x:Real32?) -> Real32? {
  return x;
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer64?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer32?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer16?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real32.
 */
function Real32(x:Integer8?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Real32.
 */
function Real32(x:String?) -> Real32? {
  if (x?) {
    return Real32(x!);
  } else {
    return nil;
  }
}

/*
 * Operators
 */
operator (x:Real32 + y:Real32) -> Real32;
operator (x:Real32 - y:Real32) -> Real32;
operator (x:Real32 * y:Real32) -> Real32;
operator (x:Real32 / y:Real32) -> Real32;
operator (+x:Real32) -> Real32;
operator (-x:Real32) -> Real32;
operator (x:Real32 > y:Real32) -> Boolean;
operator (x:Real32 < y:Real32) -> Boolean;
operator (x:Real32 >= y:Real32) -> Boolean;
operator (x:Real32 <= y:Real32) -> Boolean;
operator (x:Real32 == y:Real32) -> Boolean;
operator (x:Real32 != y:Real32) -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Real32) -> Real32 {
  cpp {{
  return std::abs(x);
  }}
}

/**
 * Power.
 */
function pow(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::powf(x, y);
  }}
}

/**
 * Modulus.
 */
function mod(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fmodf(x, y);
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::max(x, y);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::min(x, y);
  }}
}

/**
 * Does this have the value NaN?
 */
function isnan(x:Real32) -> Boolean {
  return x != x;
}
