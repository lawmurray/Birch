/**
 * 64-bit signed integer.
 */
type Integer64;

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
  return static_cast<bi::type::Integer64>(x);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Real32) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64>(x);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer32) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64>(x);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer16) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64>(x);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer8) -> Integer64 {
  cpp{{
  return static_cast<bi::type::Integer64>(x);
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Boolean) -> Integer64 {
  if x {
    return Integer64(1);
  } else {
    return Integer64(0);
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:String) -> Integer64 {
  cpp{{
  return ::atol(x.c_str());
  }}
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer64?) -> Integer64? {
  return x;
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Real64?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Real32?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer32?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer16?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:Integer8?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer64.
 */
function Integer64(x:String?) -> Integer64? {
  if (x?) {
    return Integer64(x!);
  } else {
    return nil;
  }
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
  return std::abs(x);
  }}
}

/**
 * Power.
 */
function pow(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::pow(x, y);
  }}
}

/**
 * Modulus.
 */
function mod(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return x % y;
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::max(x, y);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::min(x, y);
  }}
}
