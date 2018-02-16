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
  return std::log(x_);
  }}
}

/**
 * Natural logarithm.
 */
function log(x:Real32) -> Real32 {
  cpp {{
  return std::logf(x_);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real64) -> Real64 {
  cpp {{
  return std::log2(x_);
  }}
}

/**
 * Base-2 logarithm.
 */
function log2(x:Real32) -> Real32 {
  cpp {{
  return std::log2f(x_);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real64) -> Real64 {
  cpp {{
  return std::log10(x_);
  }}
}

/**
 * Base-10 logarithm.
 */
function log10(x:Real32) -> Real32 {
  cpp {{
  return std::log10f(x_);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real64) -> Real64 {
  cpp {{
  return std::exp(x_);
  }}
}

/**
 * Exponential.
 */
function exp(x:Real32) -> Real32 {
  cpp {{
  return std::expf(x_);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real64) -> Real64 {
  cpp {{
  return std::sqrt(x_);
  }}
}

/**
 * Square root.
 */
function sqrt(x:Real32) -> Real32 {
  cpp {{
  return std::sqrtf(x_);
  }}
}

/**
 * Power.
 */
function pow(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::pow(x_, y_);
  }}
}

/**
 * Power.
 */
function pow(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::powf(x_, y_);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real64) -> Real64 {
  cpp {{
  return std::ceil(x_);
  }}
}

/**
 * Round to smallest integer value not less than x.
 */
function ceil(x:Real32) -> Real32 {
  cpp {{
  return std::ceilf(x_);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real64) -> Real64 {
  cpp {{
  return std::floor(x_);
  }}
}

/**
 * Round to largest integer value not greater than x.
 */
function floor(x:Real32) -> Real32 {
  cpp {{
  return std::floorf(x_);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real64) -> Real64 {
  cpp {{
  return std::round(x_);
  }}
}

/**
 * Round to integer value.
 */
function round(x:Real32) -> Real32 {
  cpp {{
  return std::roundf(x_);
  }}
}

/**
 * Sine.
 */
function sin(x:Real64) -> Real64 {
  cpp {{
  return std::sin(x_);
  }}
}

/**
 * Sine.
 */
function sin(x:Real32) -> Real32 {
  cpp {{
  return std::sinf(x_);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real64) -> Real64 {
  cpp {{
  return std::cos(x_);
  }}
}

/**
 * Cosine.
 */
function cos(x:Real32) -> Real32 {
  cpp {{
  return std::cosf(x_);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real64) -> Real64 {
  cpp {{
  return std::tan(x_);
  }}
}

/**
 * Tangent.
 */
function tan(x:Real32) -> Real32 {
  cpp {{
  return std::tanf(x_);
  }}
}

/**
 * Arc sine.
 */
function asin(x:Real64) -> Real64 {
  cpp {{
  return std::asin(x_);
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
  return std::acos(x_);
  }}
}

/**
 * Arc cosine.
 */
function acos(x:Real32) -> Real32 {
  cpp {{
  return std::acosf(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real64) -> Real64 {
  cpp {{
  return std::atan(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan(x:Real32) -> Real32 {
  cpp {{
  return std::atanf(x_);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::atan2(x_, y_);
  }}
}

/**
 * Arc tangent.
 */
function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::atan2f(x_, y_);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real64) -> Real64 {
  cpp {{
  return std::sinh(x_);
  }}
}

/**
 * Hyperbolic sine.
 */
function sinh(x:Real32) -> Real32 {
  cpp {{
  return std::sinhf(x_);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real64) -> Real64 {
  cpp {{
  return std::cosh(x_);
  }}
}

/**
 * Hyperbolic cosine.
 */
function cosh(x:Real32) -> Real32 {
  cpp {{
  return std::coshf(x_);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real64) -> Real64 {
  cpp {{
  return std::tanh(x_);
  }}
}

/**
 * Hyperbolic tangent.
 */
function tanh(x:Real32) -> Real32 {
  cpp {{
  return std::tanhf(x_);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real64) -> Real64 {
  cpp {{
  return std::asinh(x_);
  }}
}

/**
 * Inverse hyperbolic sine.
 */
function asinh(x:Real32) -> Real32 {
  cpp {{
  return std::asinhf(x_);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real64) -> Real64 {
  cpp {{
  return std::acosh(x_);
  }}
}

/**
 * Inverse hyperbolic cosine.
 */
function acosh(x:Real32) -> Real32 {
  cpp {{
  return std::acoshf(x_);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real64) -> Real64 {
  cpp {{
  return std::atanh(x_);
  }}
}

/**
 * Inverse hyperbolic tangent.
 */
function atanh(x:Real32) -> Real32 {
  cpp {{
  return std::atanhf(x_);
  }}
}

/**
 * Error function.
 */
function erf(x:Real64) -> Real64 {
  cpp {{
  return std::erf(x_);
  }}
}

/**
 * Error function.
 */
function erf(x:Real32) -> Real32 {
  cpp {{
  return std::erff(x_);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real64) -> Real64 {
  cpp {{
  return std::erfc(x_);
  }}
}

/**
 * Complementary error function.
 */
function erfc(x:Real32) -> Real32 {
  cpp {{
  return std::erfcf(x_);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::copysign(x_, y_);
  }}
}

/**
 * Change sign.
 */
function copysign(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::copysignf(x_, y_);
  }}
}
