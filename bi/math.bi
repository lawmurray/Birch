cpp {{
#include <cmath>
#include <cstdint>
}}

/**
 * Built-in types
 * --------------
 */
model Boolean;
model Real64;
model Real32;
model Integer64;
model Integer32;
model Real = Real64;
model Integer = Integer64;

/**
 * Assignments
 * -----------
 */
function x:Real32 <- y:Real64 -> x {
  cpp{{
  x = y;
  }}
}

function x:Integer32 <- y:Integer64 -> x {
  cpp{{
  x = y;
  }}
}

/**
 * Operators
 * ---------
 */
/**
 * Real64 operators
 */
function x:Real64 + y:Real64 -> z:Real64 {
  cpp {{
  z = (double)x + (double)y;
  }}
}

function x:Real64 - y:Real64 -> z:Real64 {
  cpp {{
  z = (double)x - (double)y;
  }}
}

function x:Real64 * y:Real64 -> z:Real64 {
  cpp {{
  z = (double)x * (double)y;
  }}
}

function x:Real64 / y:Real64 -> z:Real64 {
  cpp {{
  z = (double)x / (double)y;
  }}
}

function +x:Real64 -> z:Real64 {
  cpp {{
  z = +(double)x;
  }}
}

function -x:Real64 -> z:Real64 {
  cpp {{
  z = -(double)x;
  }}
}

function x:Real64 > y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x > (double)y;
  }}
}

function x:Real64 < y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x < (double)y;
  }}
}

function x:Real64 >= y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x >= (double)y;
  }}
}

function x:Real64 <= y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x <= (double)y;
  }}
}

function x:Real64 == y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x == (double)y;
  }}
}

function x:Real64 != y:Real64 -> z:Boolean {
  cpp {{
  z = (double)x != (double)y;
  }}
}

/**
 * Real32 operators
 */
function x:Real32 + y:Real32 -> z:Real32 {
  cpp {{
  z = (float)x + (float)y;
  }}
}

function x:Real32 - y:Real32 -> z:Real32 {
  cpp {{
  z = (float)x - (float)y;
  }}
}

function x:Real32 * y:Real32 -> z:Real32 {
  cpp {{
  z = (float)x * (float)y;
  }}
}

function x:Real32 / y:Real32 -> z:Real32 {
  cpp {{
  z = (float)x / (float)y;
  }}
}

function +x:Real32 -> z:Real32 {
  cpp {{
  z = +(float)x;
  }}
}

function -x:Real32 -> z:Real32 {
  cpp {{
  z = -(float)x;
  }}
}

function x:Real32 > y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x > (float)y;
  }}
}

function x:Real32 < y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x < (float)y;
  }}
}

function x:Real32 >= y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x >= (float)y;
  }}
}

function x:Real32 <= y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x <= (float)y;
  }}
}

function x:Real32 == y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x == (float)y;
  }}
}

function x:Real32 != y:Real32 -> z:Boolean {
  cpp {{
  z = (float)x != (float)y;
  }}
}

/**
 * Integer64 operators
 */
function x:Integer64 + y:Integer64 -> z:Integer64 {
  cpp {{
  z = (int64_t)x + (int64_t)y;
  }}
}

function x:Integer64 - y:Integer64 -> z:Integer64 {
  cpp {{
  z = (int64_t)x - (int64_t)y;
  }}
}

function x:Integer64 * y:Integer64 -> z:Integer64 {
  cpp {{
  z = (int64_t)x * (int64_t)y;
  }}
}

function x:Integer64 / y:Integer64 -> z:Integer64 {
  cpp {{
  z = (int64_t)x / (int64_t)y;
  }}
}

function +x:Integer64 -> z:Integer64 {
  cpp {{
  z = +(int64_t)x;
  }}
}

function -x:Integer64 -> z:Integer64 {
  cpp {{
  z = -(int64_t)x;
  }}
}

function x:Integer64 > y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x > (int64_t)y;
  }}
}

function x:Integer64 < y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x < (int64_t)y;
  }}
}

function x:Integer64 >= y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x >= (int64_t)y;
  }}
}

function x:Integer64 <= y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x <= (int64_t)y;
  }}
}

function x:Integer64 == y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x == (int64_t)y;
  }}
}

function x:Integer64 != y:Integer64 -> z:Boolean {
  cpp {{
  z = (int64_t)x != (int64_t)y;
  }}
}

/**
 * Integer32 operators
 */
function x:Integer32 + y:Integer32 -> z:Integer32 {
  cpp {{
  z = (int32_t)x + (int32_t)y;
  }}
}

function x:Integer32 - y:Integer32 -> z:Integer32 {
  cpp {{
  z = (int32_t)x - (int32_t)y;
  }}
}

function x:Integer32 * y:Integer32 -> z:Integer32 {
  cpp {{
  z = (int32_t)x * (int32_t)y;
  }}
}

function x:Integer32 / y:Integer32 -> z:Integer32 {
  cpp {{
  z = (int32_t)x / (int32_t)y;
  }}
}

function +x:Integer32 -> z:Integer32 {
  cpp {{
  z = +(int32_t)x;
  }}
}

function -x:Integer32 -> z:Integer32 {
  cpp {{
  z = -(int32_t)x;
  }}
}

function x:Integer32 > y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x > (int32_t)y;
  }}
}

function x:Integer32 < y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x < (int32_t)y;
  }}
}

function x:Integer32 >= y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x >= (int32_t)y;
  }}
}

function x:Integer32 <= y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x <= (int32_t)y;
  }}
}

function x:Integer32 == y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x == (int32_t)y;
  }}
}

function x:Integer32 != y:Integer32 -> z:Boolean {
  cpp {{
  z = (int32_t)x != (int32_t)y;
  }}
}

/**
 * Boolean operators
 */
function x:Boolean && y:Boolean -> z:Boolean {
  cpp {{
  z = (bool)x && (bool)y;
  }}
}

function x:Boolean || y:Boolean -> z:Boolean {
  cpp {{
  z = (bool)x || (bool)y;
  }}
}

function !x:Boolean -> z:Boolean {
  cpp {{
  z = !(bool)x;
  }}
}

/**
 * Functions
 * ---------
 */
function abs(x:Real64) -> y:Real64 {
  cpp {{
  y = ::fabs(x);
  }}
}

function abs(x:Real32) -> y:Real32 {
  cpp {{
  y = ::fabsf(x);
  }}
}

function abs(x:Integer64) -> y:Integer64 {
  cpp {{
  y = std::abs(x);
  }}
}

function abs(x:Integer32) -> y:Integer32 {
  cpp {{
  y = std::abs(x);
  }}
}

function log(x:Real64) -> y:Real64 {
  cpp {{
  y = ::log(x);
  }}
}

function log(x:Real32) -> y:Real32 {
  cpp {{
  y = ::logf(x);
  }}
}

function nanlog(x:Real64) -> y:Real64 {
  if (isnan(x)) {
    y <- -inf;
  } else {
    y <- log(x);
  }
}

function nanlog(x:Real32) -> y:Real32 {
  if (isnan(x)) {
    y <- -inf;
  } else {
    y <- log(x);
  }
}

function exp(x:Real64) -> y:Real64 {
  cpp {{
  y = ::exp(x);
  }}
}

function exp(x:Real32) -> y:Real32 {
  cpp {{
  y = ::expf(x);
  }}
}

function nanexp(x:Real64) -> y:Real64 {
  if (isnan(x)) {
    y <- 0.0;
  } else {
    y <- exp(x);
  }
}

function nanexp(x:Real32) -> y:Real32 {
  if (isnan(x)) {
    y <- 0.0;
  } else {
    y <- exp(x);
  }
}

function max(x:Real64, y:Real64) -> z:Real64 {
  cpp {{
  z = ::fmax(x, y);
  }}
}

function max(x:Real32, y:Real32) -> z:Real32 {
  cpp {{
  z = ::fmaxf(x, y);
  }}
}

function max(x:Integer64, y:Integer64) -> z:Integer64 {
  cpp {{
  z = std::max(x, y);
  }}
}

function max(x:Integer32, y:Integer32) -> z:Integer32 {
  cpp {{
  z = std::max(x, y);
  }}
}

function min(x:Real64, y:Real64) -> z:Real64 {
  cpp {{
  z = ::fmin(x, y);
  }}
}

function min(x:Real32, y:Real32) -> z:Real32 {
  cpp {{
  z = ::fminf(x, y);
  }}
}

function min(x:Integer64, y:Integer64) -> z:Integer64 {
  cpp {{
  z = std::min(x, y);
  }}
}

function min(x:Integer32, y:Integer32) -> z:Integer32 {
  cpp {{
  z = std::min(x, y);
  }}
}

function sqrt(x:Real64) -> y:Real64 {
  cpp {{
  y = ::sqrt(x);
  }}
}

function sqrt(x:Real32) -> y:Real32 {
  cpp {{
  y = ::sqrtf(x);
  }}
}

function pow(x:Real64, y:Real64) -> z:Real64 {
  cpp {{
  z = ::pow(x, y);
  }}
}

function pow(x:Real32, y:Real32) -> z:Real32 {
  cpp {{
  z = ::powf(x, y);
  }}
}

function mod(x:Real64, y:Real64) -> z:Real64 {
  cpp {{
  z = ::fmod(x, y);
  }}
}

function mod(x:Real32, y:Real32) -> z:Real32 {
  cpp {{
  z = ::fmodf(x, y);
  }}
}

function ceil(x:Real64) -> y:Real64 {
  cpp {{
  y = ::ceil(x);
  }}
}

function ceil(x:Real32) -> y:Real32 {
  cpp {{
  y = ::ceilf(x);
  }}
}

function floor(x:Real64) -> y:Real64 {
  cpp {{
  y = ::floor(x);
  }}
}

function floor(x:Real32) -> y:Real32 {
  cpp {{
  y = ::floorf(x);
  }}
}

function round(x:Real64) -> y:Real64 {
  cpp {{
  y = ::round(x);
  }}
}

function round(x:Real32) -> y:Real32 {
  cpp {{
  y = ::roundf(x);
  }}
}

function gamma(x:Real64) -> y:Real64 {
  cpp {{
  y = ::tgamma(x);
  }}
}

function gamma(x:Real32) -> y:Real32 {
  cpp {{
  y = ::tgammaf(x);
  }}
}

function lgamma(x:Real64) -> y:Real64 {
  cpp {{
  y = ::lgamma(x);
  }}
}

function lgamma(x:Real32) -> y:Real32 {
  cpp {{
  y = ::lgammaf(x);
  }}
}

function sin(x:Real64) -> y:Real64 {
  cpp {{
  y = ::sin(x);
  }}
}

function sin(x:Real32) -> y:Real32 {
  cpp {{
  y = ::sinf(x);
  }}
}

function cos(x:Real64) -> y:Real64 {
  cpp {{
  y = ::cos(x);
  }}
}

function cos(x:Real32) -> y:Real32 {
  cpp {{
  y = ::cosf(x);
  }}
}

function tan(x:Real64) -> y:Real64 {
  cpp {{
  y = ::tan(x);
  }}
}

function tan(x:Real32) -> y:Real32 {
  cpp {{
  y = ::tanf(x);
  }}
}

function asin(x:Real64) -> y:Real64 {
  cpp {{
  y = ::asin(x);
  }}
}

function asin(x:Real32) -> y:Real32 {
 cpp {{
  y =  ::asinf(x);
  }}
}

function acos(x:Real64) -> y:Real64 {
  cpp {{
  y = ::acos(x);
  }}
}

function acos(x:Real32) -> y:Real32 {
  cpp {{
  y = ::acosf(x);
  }}
}

function atan(x:Real64) -> y:Real64 {
  cpp {{
  y = ::atan(x);
  }}
}

function atan(x:Real32) -> y:Real32 {
  cpp {{
  y = ::atanf(x);
  }}
}

function atan2(x:Real64, y:Real64) -> z:Real64 {
  cpp {{
  z = ::atan2(x, y);
  }}
}

function atan2(x:Real32, y:Real32) -> z:Real32 {
  cpp {{
  z = ::atan2f(x, y);
  }}
}

function sinh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::sinh(x);
  }}
}

function sinh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::sinhf(x);
  }}
}

function cosh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::cosh(x);
  }}
}

function cosh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::coshf(x);
  }}
}

function tanh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::tanh(x);
  }}
}

function tanh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::tanhf(x);
  }}
}

function asinh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::asinh(x);
  }}
}

function asinh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::asinhf(x);
  }}
}

function acosh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::acosh(x);
  }}
}

function acosh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::acoshf(x);
  }}
}

function atanh(x:Real64) -> y:Real64 {
  cpp {{
  y = ::atanh(x);
  }}
}

function atanh(x:Real32) -> y:Real32 {
  cpp {{
  y = ::atanhf(x);
  }}
}

function erf(x:Real64) -> y:Real64 {
  cpp {{
  y = ::erf(x);
  }}
}

function erf(x:Real32) -> y:Real32 {
  cpp {{
  y = ::erff(x);
  }}
}

function erfc(x:Real64) -> y:Real64 {
  cpp {{
  y = ::erfc(x);
  }}
}

function erfc(x:Real32) -> y:Real32 {
  cpp {{
  y = ::erfcf(x);
  }}
}

function isnan(x:Real64) -> y:Boolean {
  y <- x != x;
}

function isnan(x:Real32) -> y:Boolean {
  y <- x != x;
}

/**
 * Constants
 * ---------
 */
/**
 * $\pi$
 */
π:Real64 <- 3.1415926535897932384626433832795;

/**
 * $\infty$
 */
inf:Real64 <- 1.0/0.0;
