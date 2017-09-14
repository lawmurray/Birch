import basic;

/**
 * The gamma function.
 */
function gamma(x:Real64) -> Real64 {
  cpp {{
  return ::tgamma(x_);
  }}
}

/**
 * The gamma function.
 */
function gamma(x:Real32) -> Real32 {
  cpp {{
  return ::tgammaf(x_);
  }}
}

/**
 * Logarithm of the gamma function.
 */
function lgamma(x:Real64) -> Real64 {
  cpp {{
  return ::lgamma(x_);
  }}
}

/**
 * Logarithm of the gamma function.
 */
function lgamma(x:Real32) -> Real32 {
  cpp {{
  return ::lgammaf(x_);
  }}
}

/**
 * The beta function.
 */
function beta(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return boost::math::beta(x_, y_);
  }}
}

/**
 * The beta function.
 */
function beta(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return boost::math::beta(x_, y_);
  }}
}

/**
 * Logarithm of the beta function.
 */
function lbeta(x:Real64, y:Real64) -> Real64 {
  return lgamma(x) + lgamma(y) - lgamma(x + y);
}

/**
 * Logarithm of the beta function.
 */
function lbeta(x:Real32, y:Real32) -> Real32 {
  return lgamma(x) + lgamma(y) - lgamma(x + y);
}
