cpp{{
#include "boost/math/special_functions.hpp"
}}

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

/**
 * The binomial coefficient.
 */
function choose(x:Real64, y:Real64) -> Real64 {
  assert 0.0 <= x;
  assert 0.0 <= y;
  assert x >= y;
  
  if (y == 0.0) {
    return 1.0;
  } else {
    // see Boost binomial_coefficient function for this implementation
    return 1.0/(y*beta(y, x - y + 1.0));
  }
}

/**
 * The binomial coefficient.
 */
function choose(x:Real32, y:Real32) -> Real32 {
  assert Real32(0.0) <= x;
  assert Real32(0.0) <= y;
  assert x >= y;
  
  if (y == Real32(0.0)) {
    return Real32(1.0);
  } else {
    // see Boost binomial_coefficient function for this implementation
    return Real32(1.0)/(y*beta(y, x - y + Real32(1.0)));
  }
}

/**
 * Logarithm of the binomial coefficient.
 */
function lchoose(x:Real64, y:Real64) -> Real64 {
  assert 0.0 <= x;
  assert 0.0 <= y;
  assert x >= y;
  
  if (y == 0.0) {
    return log(1.0);
  } else {
    // see Boost binomial_coefficient function for this implementation
    return -log(y) - lbeta(y, x - y + 1.0);
  }
}

/**
 * Logarithm of the binomial coefficient.
 */
function lchoose(x:Real32, y:Real32) -> Real32 {
  assert Real32(0.0) <= x;
  assert Real32(0.0) <= y;
  assert x >= y;
  
  if (y == Real32(0.0)) {
    return log(Real32(1.0));
  } else {
    // see Boost binomial_coefficient function for this implementation
    return -log(y) - lbeta(y, x - y + Real32(1.0));
  }
}

/**
 * Rising factorial.
 */
function rising(x:Real64, i:Integer32) -> Real64 {
  cpp{{
  return boost::math::rising_factorial(x_, i_);
  }}
}

/**
 * Rising factorial.
 */
function rising(x:Real32, i:Integer32) -> Real32 {
  cpp{{
  return boost::math::rising_factorial(x_, i_);
  }}
}

/**
 * Logarithm of the rising factorial.
 */
function lrising(x:Real64, i:Real64) -> Real64 {
  return lgamma(x + i) - lgamma(x);
}

/**
 * Logarithm of the rising factorial.
 */
function lrising(x:Real32, i:Real32) -> Real32 {
  return lgamma(x + i) - lgamma(x);
}
