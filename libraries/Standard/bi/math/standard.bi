/**
 * $\pi$
 */
π:Real64 <- 3.1415926535897932384626433832795;

/**
 * $\infty$
 */
inf:Real64 <- 1.0/0.0;

/**
 * NaN.
 */
nan:Real64 <- 0.0/0.0;

/**
 * Natural logarithm.
 */
function log(x:Real64) -> Real64 {
  cpp {{
  return ::log(x);
  }}
}

/**
 * Natural logarithm.
 */
function log(x:Real32) -> Real32 {
  cpp {{
  return ::logf(x);
  }}
}

/**
 * Natural logarithm of one plus argument.
 */
function log1p(x:Real64) -> Real64 {
  cpp {{
  return ::log1p(x);
  }}
}

/**
 * Natural logarithm of one plus argument.
 */
function log1p(x:Real32) -> Real32 {
  cpp {{
  return ::log1pf(x);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real64) -> Real64 {
  cpp {{
  return ::log2(x);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real32) -> Real32 {
  cpp {{
  return ::log2f(x);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real64) -> Real64 {
  cpp {{
  return ::log10(x);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real32) -> Real32 {
  cpp {{
  return ::log10f(x);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real64) -> Real64 {
  cpp {{
  return ::exp(x);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real32) -> Real32 {
  cpp {{
  return ::expf(x);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real64) -> Real64 {
  cpp {{
  return ::sqrt(x);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real32) -> Real32 {
  cpp {{
  return ::sqrtf(x);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real64) -> Real64 {
  cpp {{
  return ::ceil(x);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real32) -> Real32 {
  cpp {{
  return ::ceilf(x);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real64) -> Real64 {
  cpp {{
  return ::floor(x);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real32) -> Real32 {
  cpp {{
  return ::floorf(x);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real64) -> Real64 {
  cpp {{
  return ::round(x);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real32) -> Real32 {
  cpp {{
  return ::roundf(x);
  }}
}

/**
 * Sine.
 */
function sin(x:Real64) -> Real64 {
  cpp {{
  return ::sin(x);
  }}
}

/**
 * Sine.
 */
function sin(x:Real32) -> Real32 {
  cpp {{
  return ::sinf(x);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real64) -> Real64 {
  cpp {{
  return ::cos(x);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real32) -> Real32 {
  cpp {{
  return ::cosf(x);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real64) -> Real64 {
  cpp {{
  return ::tan(x);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real32) -> Real32 {
  cpp {{
  return ::tanf(x);
  }}
}

/**
 * Arc sine.
 */
function asin(x:Real64) -> Real64 {
  cpp {{
  return ::asin(x);
  }}
}

/**
 * Arc sine.
 */
function asin(x:Real32) -> Real32 {
 cpp {{
  return  ::asinf(x);
  }}
}

/**
 * Arc cosine.
 */
function acos(x:Real64) -> Real64 {
  cpp {{
  return ::acos(x);
  }}
}

/**
 * Arc cosine.
 */
function acos(x:Real32) -> Real32 {
  cpp {{
  return ::acosf(x);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real64) -> Real64 {
  cpp {{
  return ::atan(x);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real32) -> Real32 {
  cpp {{
  return ::atanf(x);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::atan2(x, y);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::atan2f(x, y);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real64) -> Real64 {
  cpp {{
  return ::sinh(x);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real32) -> Real32 {
  cpp {{
  return ::sinhf(x);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real64) -> Real64 {
  cpp {{
  return ::cosh(x);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real32) -> Real32 {
  cpp {{
  return ::coshf(x);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real64) -> Real64 {
  cpp {{
  return ::tanh(x);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real32) -> Real32 {
  cpp {{
  return ::tanhf(x);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real64) -> Real64 {
  cpp {{
  return ::asinh(x);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real32) -> Real32 {
  cpp {{
  return ::asinhf(x);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real64) -> Real64 {
  cpp {{
  return ::acosh(x);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real32) -> Real32 {
  cpp {{
  return ::acoshf(x);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real64) -> Real64 {
  cpp {{
  return ::atanh(x);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real32) -> Real32 {
  cpp {{
  return ::atanhf(x);
  }}
}

/**
 * Error function.
 */
function erf(x:Real64) -> Real64 {
  cpp {{
  return ::erf(x);
  }}
}

/**
 * Error function.
 */
function erf(x:Real32) -> Real32 {
  cpp {{
  return ::erff(x);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real64) -> Real64 {
  cpp {{
  return ::erfc(x);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real32) -> Real32 {
  cpp {{
  return ::erfcf(x);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::copysign(x, y);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::copysignf(x, y);
  }}
}

/**
 * Rectify.
 */
function rectify(x:Real64) -> Real64 {
  return max(0.0, x);
}

/**
 * Rectify.
 */
function rectify(x:Real32) -> Real32 {
  return max(Real32(0.0), x);
}
