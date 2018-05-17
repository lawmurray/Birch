/**
 * $\pi$
 */
π:Real64 <- 3.1415926535897932384626433832795;

/**
 * $\infty$
 */
inf:Real64 <- 1.0/0.0;

/**
 * Natural logarithm.
 */
function log(x:Real64) -> Real64 {
  cpp {{
  return ::log(x_);
  }}
}

/**
 * Natural logarithm.
 */
function log(x:Real32) -> Real32 {
  cpp {{
  return ::logf(x_);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real64) -> Real64 {
  cpp {{
  return ::log2(x_);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real32) -> Real32 {
  cpp {{
  return ::log2f(x_);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real64) -> Real64 {
  cpp {{
  return ::log10(x_);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real32) -> Real32 {
  cpp {{
  return ::log10f(x_);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real64) -> Real64 {
  cpp {{
  return ::exp(x_);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real32) -> Real32 {
  cpp {{
  return ::expf(x_);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real64) -> Real64 {
  cpp {{
  return ::sqrt(x_);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real32) -> Real32 {
  cpp {{
  return ::sqrtf(x_);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real64) -> Real64 {
  cpp {{
  return ::ceil(x_);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real32) -> Real32 {
  cpp {{
  return ::ceilf(x_);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real64) -> Real64 {
  cpp {{
  return ::floor(x_);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real32) -> Real32 {
  cpp {{
  return ::floorf(x_);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real64) -> Real64 {
  cpp {{
  return ::round(x_);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real32) -> Real32 {
  cpp {{
  return ::roundf(x_);
  }}
}

/**
 * Sine.
 */
function sin(x:Real64) -> Real64 {
  cpp {{
  return ::sin(x_);
  }}
}

/**
 * Sine.
 */
function sin(x:Real32) -> Real32 {
  cpp {{
  return ::sinf(x_);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real64) -> Real64 {
  cpp {{
  return ::cos(x_);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real32) -> Real32 {
  cpp {{
  return ::cosf(x_);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real64) -> Real64 {
  cpp {{
  return ::tan(x_);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real32) -> Real32 {
  cpp {{
  return ::tanf(x_);
  }}
}

/**
 * Arc sine.
 */
function asin(x:Real64) -> Real64 {
  cpp {{
  return ::asin(x_);
  }}
}

/**
 * Arc sine.
 */
function asin(x:Real32) -> Real32 {
 cpp {{
  return  ::asinf(x_);
  }}
}

/**
 * Arc cosine.
 */
function acos(x:Real64) -> Real64 {
  cpp {{
  return ::acos(x_);
  }}
}

/**
 * Arc cosine.
 */
function acos(x:Real32) -> Real32 {
  cpp {{
  return ::acosf(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real64) -> Real64 {
  cpp {{
  return ::atan(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real32) -> Real32 {
  cpp {{
  return ::atanf(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::atan2(x_, y_);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::atan2f(x_, y_);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real64) -> Real64 {
  cpp {{
  return ::sinh(x_);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real32) -> Real32 {
  cpp {{
  return ::sinhf(x_);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real64) -> Real64 {
  cpp {{
  return ::cosh(x_);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real32) -> Real32 {
  cpp {{
  return ::coshf(x_);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real64) -> Real64 {
  cpp {{
  return ::tanh(x_);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real32) -> Real32 {
  cpp {{
  return ::tanhf(x_);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real64) -> Real64 {
  cpp {{
  return ::asinh(x_);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real32) -> Real32 {
  cpp {{
  return ::asinhf(x_);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real64) -> Real64 {
  cpp {{
  return ::acosh(x_);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real32) -> Real32 {
  cpp {{
  return ::acoshf(x_);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real64) -> Real64 {
  cpp {{
  return ::atanh(x_);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real32) -> Real32 {
  cpp {{
  return ::atanhf(x_);
  }}
}

/**
 * Error function.
 */
function erf(x:Real64) -> Real64 {
  cpp {{
  return ::erf(x_);
  }}
}

/**
 * Error function.
 */
function erf(x:Real32) -> Real32 {
  cpp {{
  return ::erff(x_);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real64) -> Real64 {
  cpp {{
  return ::erfc(x_);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real32) -> Real32 {
  cpp {{
  return ::erfcf(x_);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::copysign(x_, y_);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::copysignf(x_, y_);
  }}
}
