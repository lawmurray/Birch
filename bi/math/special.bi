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
  return 1.0/(y*beta(y, x - y + 1.0));
}

/**
 * The binomial coefficient.
 */
function choose(x:Real32, y:Real32) -> Real32 {
  assert Real32(0.0) <= x;
  assert Real32(0.0) <= y;
  return Real32(1.0)/(y*beta(y, x - y + Real32(1.0)));
}

/**
 * Logarithm of the binomial coefficient.
 */
function lchoose(x:Real64, y:Real64) -> Real64 {
  assert 0.0 <= x;
  assert 0.0 <= y;
  return -log(y) - lbeta(y, x - y + 1.0);
}

/**
 * Logarithm of the binomial coefficient.
 */
function lchoose(x:Real32, y:Real32) -> Real32 {
  assert Real32(0.0) <= x;
  assert Real32(0.0) <= y;
  return -log(y) - lbeta(y, x - y + Real32(1.0));
}
