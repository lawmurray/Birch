import basic;

/**
 * $\pi$
 */
π:Real64 <- 3.1415926535897932384626433832795;

/**
 * $\infty$
 */
inf:Real64 <- 1.0/0.0;

function log(x:Real64) -> Real64 {
  cpp {{
  return ::log(x_);
  }}
}

function log(x:Real32) -> Real32 {
  cpp {{
  return ::logf(x_);
  }}
}

function exp(x:Real64) -> Real64 {
  cpp {{
  return ::exp(x_);
  }}
}

function exp(x:Real32) -> Real32 {
  cpp {{
  return ::expf(x_);
  }}
}

function sqrt(x:Real64) -> Real64 {
  cpp {{
  return ::sqrt(x_);
  }}
}

function sqrt(x:Real32) -> Real32 {
  cpp {{
  return ::sqrtf(x_);
  }}
}

function pow(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::pow(x_, y_);
  }}
}

function pow(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::powf(x_, y_);
  }}
}

function mod(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmod(x_, y_);
  }}
}

function mod(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fmodf(x_, y_);
  }}
}

function ceil(x:Real64) -> Real64 {
  cpp {{
  return ::ceil(x_);
  }}
}

function ceil(x:Real32) -> Real32 {
  cpp {{
  return ::ceilf(x_);
  }}
}

function floor(x:Real64) -> Real64 {
  cpp {{
  return ::floor(x_);
  }}
}

function floor(x:Real32) -> Real32 {
  cpp {{
  return ::floorf(x_);
  }}
}

function round(x:Real64) -> Real64 {
  cpp {{
  return ::round(x_);
  }}
}

function round(x:Real32) -> Real32 {
  cpp {{
  return ::roundf(x_);
  }}
}

function gamma(x:Real64) -> Real64 {
  cpp {{
  return ::tgamma(x_);
  }}
}

function gamma(x:Real32) -> Real32 {
  cpp {{
  return ::tgammaf(x_);
  }}
}

function lgamma(x:Real64) -> Real64 {
  cpp {{
  return ::lgamma(x_);
  }}
}

function lgamma(x:Real32) -> Real32 {
  cpp {{
  return ::lgammaf(x_);
  }}
}

function sin(x:Real64) -> Real64 {
  cpp {{
  return ::sin(x_);
  }}
}

function sin(x:Real32) -> Real32 {
  cpp {{
  return ::sinf(x_);
  }}
}

function cos(x:Real64) -> Real64 {
  cpp {{
  return ::cos(x_);
  }}
}

function cos(x:Real32) -> Real32 {
  cpp {{
  return ::cosf(x_);
  }}
}

function tan(x:Real64) -> Real64 {
  cpp {{
  return ::tan(x_);
  }}
}

function tan(x:Real32) -> Real32 {
  cpp {{
  return ::tanf(x_);
  }}
}

function asin(x:Real64) -> Real64 {
  cpp {{
  return ::asin(x_);
  }}
}

function asin(x:Real32) -> Real32 {
 cpp {{
  return  ::asinf(x_);
  }}
}

function acos(x:Real64) -> Real64 {
  cpp {{
  return ::acos(x_);
  }}
}

function acos(x:Real32) -> Real32 {
  cpp {{
  return ::acosf(x_);
  }}
}

function atan(x:Real64) -> Real64 {
  cpp {{
  return ::atan(x_);
  }}
}

function atan(x:Real32) -> Real32 {
  cpp {{
  return ::atanf(x_);
  }}
}

function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::atan2(x_, y_);
  }}
}

function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::atan2f(x_, y_);
  }}
}

function sinh(x:Real64) -> Real64 {
  cpp {{
  return ::sinh(x_);
  }}
}

function sinh(x:Real32) -> Real32 {
  cpp {{
  return ::sinhf(x_);
  }}
}

function cosh(x:Real64) -> Real64 {
  cpp {{
  return ::cosh(x_);
  }}
}

function cosh(x:Real32) -> Real32 {
  cpp {{
  return ::coshf(x_);
  }}
}

function tanh(x:Real64) -> Real64 {
  cpp {{
  return ::tanh(x_);
  }}
}

function tanh(x:Real32) -> Real32 {
  cpp {{
  return ::tanhf(x_);
  }}
}

function asinh(x:Real64) -> Real64 {
  cpp {{
  return ::asinh(x_);
  }}
}

function asinh(x:Real32) -> Real32 {
  cpp {{
  return ::asinhf(x_);
  }}
}

function acosh(x:Real64) -> Real64 {
  cpp {{
  return ::acosh(x_);
  }}
}

function acosh(x:Real32) -> Real32 {
  cpp {{
  return ::acoshf(x_);
  }}
}

function atanh(x:Real64) -> Real64 {
  cpp {{
  return ::atanh(x_);
  }}
}

function atanh(x:Real32) -> Real32 {
  cpp {{
  return ::atanhf(x_);
  }}
}

function erf(x:Real64) -> Real64 {
  cpp {{
  return ::erf(x_);
  }}
}

function erf(x:Real32) -> Real32 {
  cpp {{
  return ::erff(x_);
  }}
}

function erfc(x:Real64) -> Real64 {
  cpp {{
  return ::erfc(x_);
  }}
}

function erfc(x:Real32) -> Real32 {
  cpp {{
  return ::erfcf(x_);
  }}
}

function copysign(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::copysign(x_, y_);
  }}
}

function copysign(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::copysignf(x_, y_);
  }}
}
