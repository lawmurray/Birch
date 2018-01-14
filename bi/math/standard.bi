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
  return std::log(x_);
  }}
}

function log(x:Real32) -> Real32 {
  cpp {{
  return std::logf(x_);
  }}
}

function log2(x:Real64) -> Real64 {
  cpp {{
  return std::log2(x_);
  }}
}

function log2(x:Real32) -> Real32 {
  cpp {{
  return std::log2f(x_);
  }}
}

function log10(x:Real64) -> Real64 {
  cpp {{
  return std::log10(x_);
  }}
}

function log10(x:Real32) -> Real32 {
  cpp {{
  return std::log10f(x_);
  }}
}

function exp(x:Real64) -> Real64 {
  cpp {{
  return std::exp(x_);
  }}
}

function exp(x:Real32) -> Real32 {
  cpp {{
  return std::expf(x_);
  }}
}

function sqrt(x:Real64) -> Real64 {
  cpp {{
  return std::sqrt(x_);
  }}
}

function sqrt(x:Real32) -> Real32 {
  cpp {{
  return std::sqrtf(x_);
  }}
}

function pow(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::pow(x_, y_);
  }}
}

function pow(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::powf(x_, y_);
  }}
}

function ceil(x:Real64) -> Real64 {
  cpp {{
  return std::ceil(x_);
  }}
}

function ceil(x:Real32) -> Real32 {
  cpp {{
  return std::ceilf(x_);
  }}
}

function floor(x:Real64) -> Real64 {
  cpp {{
  return std::floor(x_);
  }}
}

function floor(x:Real32) -> Real32 {
  cpp {{
  return std::floorf(x_);
  }}
}

function round(x:Real64) -> Real64 {
  cpp {{
  return std::round(x_);
  }}
}

function round(x:Real32) -> Real32 {
  cpp {{
  return std::roundf(x_);
  }}
}

function sin(x:Real64) -> Real64 {
  cpp {{
  return std::sin(x_);
  }}
}

function sin(x:Real32) -> Real32 {
  cpp {{
  return std::sinf(x_);
  }}
}

function cos(x:Real64) -> Real64 {
  cpp {{
  return std::cos(x_);
  }}
}

function cos(x:Real32) -> Real32 {
  cpp {{
  return std::cosf(x_);
  }}
}

function tan(x:Real64) -> Real64 {
  cpp {{
  return std::tan(x_);
  }}
}

function tan(x:Real32) -> Real32 {
  cpp {{
  return std::tanf(x_);
  }}
}

function asin(x:Real64) -> Real64 {
  cpp {{
  return std::asin(x_);
  }}
}

function asin(x:Real32) -> Real32 {
 cpp {{
  return  ::asinf(x_);
  }}
}

function acos(x:Real64) -> Real64 {
  cpp {{
  return std::acos(x_);
  }}
}

function acos(x:Real32) -> Real32 {
  cpp {{
  return std::acosf(x_);
  }}
}

function atan(x:Real64) -> Real64 {
  cpp {{
  return std::atan(x_);
  }}
}

function atan(x:Real32) -> Real32 {
  cpp {{
  return std::atanf(x_);
  }}
}

function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::atan2(x_, y_);
  }}
}

function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::atan2f(x_, y_);
  }}
}

function sinh(x:Real64) -> Real64 {
  cpp {{
  return std::sinh(x_);
  }}
}

function sinh(x:Real32) -> Real32 {
  cpp {{
  return std::sinhf(x_);
  }}
}

function cosh(x:Real64) -> Real64 {
  cpp {{
  return std::cosh(x_);
  }}
}

function cosh(x:Real32) -> Real32 {
  cpp {{
  return std::coshf(x_);
  }}
}

function tanh(x:Real64) -> Real64 {
  cpp {{
  return std::tanh(x_);
  }}
}

function tanh(x:Real32) -> Real32 {
  cpp {{
  return std::tanhf(x_);
  }}
}

function asinh(x:Real64) -> Real64 {
  cpp {{
  return std::asinh(x_);
  }}
}

function asinh(x:Real32) -> Real32 {
  cpp {{
  return std::asinhf(x_);
  }}
}

function acosh(x:Real64) -> Real64 {
  cpp {{
  return std::acosh(x_);
  }}
}

function acosh(x:Real32) -> Real32 {
  cpp {{
  return std::acoshf(x_);
  }}
}

function atanh(x:Real64) -> Real64 {
  cpp {{
  return std::atanh(x_);
  }}
}

function atanh(x:Real32) -> Real32 {
  cpp {{
  return std::atanhf(x_);
  }}
}

function erf(x:Real64) -> Real64 {
  cpp {{
  return std::erf(x_);
  }}
}

function erf(x:Real32) -> Real32 {
  cpp {{
  return std::erff(x_);
  }}
}

function erfc(x:Real64) -> Real64 {
  cpp {{
  return std::erfc(x_);
  }}
}

function erfc(x:Real32) -> Real32 {
  cpp {{
  return std::erfcf(x_);
  }}
}

function copysign(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return std::copysign(x_, y_);
  }}
}

function copysign(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return std::copysignf(x_, y_);
  }}
}
