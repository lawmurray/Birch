cpp {{
#include <cmath>
#include <cstdint>
}}

/**
 * Built-in types
 * --------------
 */
type Boolean;
type Real64;
type Real32;
type Integer64;
type Integer32;
type Real = Real64;
type Integer = Integer64;

/**
 * Conversions
 * -----------
 */
function Real64(x:Real32) -> Real64 {
  cpp{{
  return x;
  }}
}

function Real32(x:Real64) -> Real32 {
  cpp{{
  return static_cast<float>(x);
  }}
}

function Integer64(x:Integer32) -> Integer64 {
  cpp{{
  return x;
  }}
}

function Integer32(x:Integer64) -> Integer32 {
  cpp{{
  return static_cast<int32_t>(x);
  }}
}

/**
 * Operators
 * ---------
 */
/**
 * Real64 operators
 */
function (x:Real64 + y:Real64) -> Real64;
function (x:Real64 - y:Real64) -> Real64;
function (x:Real64 * y:Real64) -> Real64;
function (x:Real64 / y:Real64) -> Real64;
function (+x:Real64) -> Real64;
function (-x:Real64) -> Real64;
function (x:Real64 > y:Real64) -> Boolean;
function (x:Real64 < y:Real64) -> Boolean;
function (x:Real64 >= y:Real64) -> Boolean;
function (x:Real64 <= y:Real64) -> Boolean;
function (x:Real64 == y:Real64) -> Boolean;
function (x:Real64 != y:Real64) -> Boolean;

/**
 * Real32 operators
 */
function (x:Real32 + y:Real32) -> Real32;
function (x:Real32 - y:Real32) -> Real32;
function (x:Real32 * y:Real32) -> Real32;
function (x:Real32 / y:Real32) -> Real32;
function (+x:Real32) -> Real32;
function (-x:Real32) -> Real32;
function (x:Real32 > y:Real32) -> Boolean;
function (x:Real32 < y:Real32) -> Boolean;
function (x:Real32 >= y:Real32) -> Boolean;
function (x:Real32 <= y:Real32) -> Boolean;
function (x:Real32 == y:Real32) -> Boolean;
function (x:Real32 != y:Real32) -> Boolean;

/**
 * Integer64 operators
 */
function (x:Integer64 + y:Integer64) -> Integer64;
function (x:Integer64 - y:Integer64) -> Integer64;
function (x:Integer64 * y:Integer64) -> Integer64;
function (x:Integer64 / y:Integer64) -> Integer64;
function (+x:Integer64) -> Integer64;
function (-x:Integer64) -> Integer64;
function (x:Integer64 > y:Integer64) -> Boolean;
function (x:Integer64 < y:Integer64) -> Boolean;
function (x:Integer64 >= y:Integer64) -> Boolean;
function (x:Integer64 <= y:Integer64) -> Boolean;
function (x:Integer64 == y:Integer64) -> Boolean;
function (x:Integer64 != y:Integer64) -> Boolean;

/**
 * Integer32 operators
 */
function (x:Integer32 + y:Integer32) -> Integer32;
function (x:Integer32 - y:Integer32) -> Integer32;
function (x:Integer32 * y:Integer32) -> Integer32;
function (x:Integer32 / y:Integer32) -> Integer32;
function (+x:Integer32) -> Integer32;
function (-x:Integer32) -> Integer32;
function (x:Integer32 > y:Integer32) -> Boolean;
function (x:Integer32 < y:Integer32) -> Boolean;
function (x:Integer32 >= y:Integer32) -> Boolean;
function (x:Integer32 <= y:Integer32) -> Boolean;
function (x:Integer32 == y:Integer32) -> Boolean;
function (x:Integer32 != y:Integer32) -> Boolean;

/**
 * Boolean operators
 */
function (x:Boolean && y:Boolean) -> Boolean;
function (x:Boolean || y:Boolean) -> Boolean;
function (!x:Boolean) -> Boolean;

/**
 * Functions
 * ---------
 */
function abs(x:Real64) -> Real64 {
  cpp {{
  return ::fabs(x);
  }}
}

function abs(x:Real32) -> Real32 {
  cpp {{
  return ::fabsf(x);
  }}
}

function abs(x:Integer64) -> Integer64 {
  cpp {{
  return std::abs(x);
  }}
}

function abs(x:Integer32) -> Integer32 {
  cpp {{
  return std::abs(x);
  }}
}

function log(x:Real64) -> Real64 {
  cpp {{
  return ::log(x);
  }}
}

function log(x:Real32) -> Real32 {
  cpp {{
  return ::logf(x);
  }}
}

function exp(x:Real64) -> Real64 {
  cpp {{
  return ::exp(x);
  }}
}

function exp(x:Real32) -> Real32 {
  cpp {{
  return ::expf(x);
  }}
}

function max(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmax(x, y);
  }}
}

function max(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fmaxf(x, y);
  }}
}

function max(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::max(x, y);
  }}
}

function max(x:Integer32, y:Integer32) -> Integer32 {
  cpp {{
  return std::max(x, y);
  }}
}

function min(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmin(x, y);
  }}
}

function min(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fminf(x, y);
  }}
}

function min(x:Integer64, y:Integer64) -> Integer64 {
  cpp {{
  return std::min(x, y);
  }}
}

function min(x:Integer32, y:Integer32) -> Integer32 {
  cpp {{
  return std::min(x, y);
  }}
}

function sqrt(x:Real64) -> Real64 {
  cpp {{
  return ::sqrt(x);
  }}
}

function sqrt(x:Real32) -> Real32 {
  cpp {{
  return ::sqrtf(x);
  }}
}

function pow(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::pow(x, y);
  }}
}

function pow(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::powf(x, y);
  }}
}

function mod(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::fmod(x, y);
  }}
}

function mod(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::fmodf(x, y);
  }}
}

function ceil(x:Real64) -> Real64 {
  cpp {{
  return ::ceil(x);
  }}
}

function ceil(x:Real32) -> Real32 {
  cpp {{
  return ::ceilf(x);
  }}
}

function floor(x:Real64) -> Real64 {
  cpp {{
  return ::floor(x);
  }}
}

function floor(x:Real32) -> Real32 {
  cpp {{
  return ::floorf(x);
  }}
}

function round(x:Real64) -> Real64 {
  cpp {{
  return ::round(x);
  }}
}

function round(x:Real32) -> Real32 {
  cpp {{
  return ::roundf(x);
  }}
}

function gamma(x:Real64) -> Real64 {
  cpp {{
  return ::tgamma(x);
  }}
}

function gamma(x:Real32) -> Real32 {
  cpp {{
  return ::tgammaf(x);
  }}
}

function lgamma(x:Real64) -> Real64 {
  cpp {{
  return ::lgamma(x);
  }}
}

function lgamma(x:Real32) -> Real32 {
  cpp {{
  return ::lgammaf(x);
  }}
}

function sin(x:Real64) -> Real64 {
  cpp {{
  return ::sin(x);
  }}
}

function sin(x:Real32) -> Real32 {
  cpp {{
  return ::sinf(x);
  }}
}

function cos(x:Real64) -> Real64 {
  cpp {{
  return ::cos(x);
  }}
}

function cos(x:Real32) -> Real32 {
  cpp {{
  return ::cosf(x);
  }}
}

function tan(x:Real64) -> Real64 {
  cpp {{
  return ::tan(x);
  }}
}

function tan(x:Real32) -> Real32 {
  cpp {{
  return ::tanf(x);
  }}
}

function asin(x:Real64) -> Real64 {
  cpp {{
  return ::asin(x);
  }}
}

function asin(x:Real32) -> Real32 {
 cpp {{
  return  ::asinf(x);
  }}
}

function acos(x:Real64) -> Real64 {
  cpp {{
  return ::acos(x);
  }}
}

function acos(x:Real32) -> Real32 {
  cpp {{
  return ::acosf(x);
  }}
}

function atan(x:Real64) -> Real64 {
  cpp {{
  return ::atan(x);
  }}
}

function atan(x:Real32) -> Real32 {
  cpp {{
  return ::atanf(x);
  }}
}

function atan2(x:Real64, y:Real64) -> Real64 {
  cpp {{
  return ::atan2(x, y);
  }}
}

function atan2(x:Real32, y:Real32) -> Real32 {
  cpp {{
  return ::atan2f(x, y);
  }}
}

function sinh(x:Real64) -> Real64 {
  cpp {{
  return ::sinh(x);
  }}
}

function sinh(x:Real32) -> Real32 {
  cpp {{
  return ::sinhf(x);
  }}
}

function cosh(x:Real64) -> Real64 {
  cpp {{
  return ::cosh(x);
  }}
}

function cosh(x:Real32) -> Real32 {
  cpp {{
  return ::coshf(x);
  }}
}

function tanh(x:Real64) -> Real64 {
  cpp {{
  return ::tanh(x);
  }}
}

function tanh(x:Real32) -> Real32 {
  cpp {{
  return ::tanhf(x);
  }}
}

function asinh(x:Real64) -> Real64 {
  cpp {{
  return ::asinh(x);
  }}
}

function asinh(x:Real32) -> Real32 {
  cpp {{
  return ::asinhf(x);
  }}
}

function acosh(x:Real64) -> Real64 {
  cpp {{
  return ::acosh(x);
  }}
}

function acosh(x:Real32) -> Real32 {
  cpp {{
  return ::acoshf(x);
  }}
}

function atanh(x:Real64) -> Real64 {
  cpp {{
  return ::atanh(x);
  }}
}

function atanh(x:Real32) -> Real32 {
  cpp {{
  return ::atanhf(x);
  }}
}

function erf(x:Real64) -> Real64 {
  cpp {{
  return ::erf(x);
  }}
}

function erf(x:Real32) -> Real32 {
  cpp {{
  return ::erff(x);
  }}
}

function erfc(x:Real64) -> Real64 {
  cpp {{
  return ::erfc(x);
  }}
}

function erfc(x:Real32) -> Real32 {
  cpp {{
  return ::erfcf(x);
  }}
}

function isnan(x:Real64) -> Boolean {
  return x != x;
}

function isnan(x:Real32) -> Boolean {
  return x != x;
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
