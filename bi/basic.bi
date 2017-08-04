hpp{{
namespace bi {
  namespace type {
using Boolean_ = bool;
using Real64_ = double;
using Real32_ = float;
using Integer64_ = int64_t;
using Integer32_ = int32_t;
using String_ = std::string;
  }
}
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
type String;

type Real = Real64;
type Integer = Integer64;

/**
 * Basic conversions
 * -----------------
 */
function Boolean(s:String) -> Boolean {
  cpp{{
  return ::atoi(s_.c_str());
  }}
}

function Real64(s:String) -> Real64 {
  cpp{{
  return ::strtod(s_.c_str(), nullptr);
  }}
}

function Real32(s:String) -> Real32 {
  cpp{{
  return ::strtof(s_.c_str(), nullptr);
  }}
}

function Integer64(s:String) -> Integer64 {
  cpp{{
  return ::atol(s_.c_str());
  }}
}

function Integer32(s:String) -> Integer32 {
  cpp{{
  return ::atoi(s_.c_str());
  }}
}

function Real(s:String) -> Real {
  cpp{{
  return ::strtod(s_.c_str(), nullptr);
  }}
}

function Real(x:Integer) -> Real {
  cpp{{
  return static_cast<bi::type::Real_>(x_);
  }}
}

function Integer(x:Real) -> Integer {
  cpp{{
  return static_cast<bi::type::Integer_>(x_);
  }}
}

function Integer(s:String) -> Integer {
  cpp{{
  return ::atol(s_.c_str());
  }}
}

function String(s:String) -> String {
  return s;
}
