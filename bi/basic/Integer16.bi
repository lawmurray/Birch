/**
 * 16-bit signed integer.
 */
type Integer16 < Integer32;

/**
 * Convert to Integer16.
 */
function Integer16(x:Real64) -> Integer16 {
  cpp{{
  return static_cast<bi::type::Integer16_>(x_);
  }}
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Real32) -> Integer16 {
  cpp{{
  return static_cast<bi::type::Integer16_>(x_);
  }}
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer64) -> Integer16 {
  cpp{{
  return static_cast<bi::type::Integer16_>(x_);
  }}
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer32) -> Integer16 {
  cpp{{
  return static_cast<bi::type::Integer16_>(x_);
  }}
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer16) -> Integer16 {
  return x;
}

/**
 * Convert to Integer16.
 */
function Integer16(x:String) -> Integer16 {
  cpp{{
  return ::atoi(x_.c_str());
  }}
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Real64?) -> Integer16? {
  if (x?) {
    return Integer16(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Real32?) -> Integer16? {
  if (x?) {
    return Integer16(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer64?) -> Integer16? {
  if (x?) {
    return Integer16(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer32?) -> Integer16? {
  if (x?) {
    return Integer16(x!);
  } else {
    return nil;
  }
}

/**
 * Convert to Integer16.
 */
function Integer16(x:Integer16?) -> Integer16? {
  return x;
}

/**
 * Convert to Integer16.
 */
function Integer16(x:String?) -> Integer16? {
  if (x?) {
    return Integer16(x!);
  } else {
    return nil;
  }
}

/*
 * Operators
 */
operator (x:Integer16 + y:Integer16) -> Integer16;
operator (x:Integer16 - y:Integer16) -> Integer16;
operator (x:Integer16 * y:Integer16) -> Integer16;
operator (x:Integer16 / y:Integer16) -> Integer16;
operator (+x:Integer16) -> Integer16;
operator (-x:Integer16) -> Integer16;
operator (x:Integer16 > y:Integer16) -> Boolean;
operator (x:Integer16 < y:Integer16) -> Boolean;
operator (x:Integer16 >= y:Integer16) -> Boolean;
operator (x:Integer16 <= y:Integer16) -> Boolean;
operator (x:Integer16 == y:Integer16) -> Boolean;
operator (x:Integer16 != y:Integer16) -> Boolean;

/**
 * Absolute value.
 */
function abs(x:Integer16) -> Integer16 {
  cpp {{
  return std::abs(x_);
  }}
}

/**
 * Modulus.
 */
function mod(x:Integer16, y:Integer16) -> Integer16 {
  cpp {{
  return x_ % y_;
  }}
}

/**
 * Maximum of two values.
 */
function max(x:Integer16, y:Integer16) -> Integer16 {
  cpp {{
  return std::max(x_, y_);
  }}
}

/**
 * Minimum of two values.
 */
function min(x:Integer16, y:Integer16) -> Integer16 {
  cpp {{
  return std::min(x_, y_);
  }}
}

/**
 * Round an integer up to the next power of two. Zero and negative integers
 * return zero.
 */
function pow2(x:Integer16) -> Integer16 {
  if (x < 0) {
    return Integer16(0);
  } else {
    y:Integer16 <- x/2;
    z:Integer16 <- 1;
    while (y > 0) {
      y <- y/Integer16(2);
      z <- z*Integer16(2);
    }
    return z;
  }
}
