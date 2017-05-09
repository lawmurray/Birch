cpp{{
#include <cstdlib>
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
type String;

/**
 * Basic conversions
 * -----------------
 */
function Boolean(s:String) -> Boolean {
  cpp{{
  return ::atoi(s.c_str());
  }}
}

function Real(s:String) -> Real {
  cpp{{
  return ::strtof(s.c_str(), nullptr);
  }}
}

function Integer(s:String) -> Integer {
  cpp{{
  return ::atol(s.c_str());
  }}
}

function Real(x:Integer) -> Real {
  cpp{{
  return static_cast<bi::type::Real>(x);
  }}
}

/**
 * String assignment
 * -----------------
 */
function x:Boolean <- s:String {
  cpp{{
  x = ::atoi(s.c_str());
  }}
}

function x:Real64 <- s:String {
  cpp{{
  x = ::strtof(s.c_str(), nullptr);
  }}
}

function x:Real32 <- s:String {
  cpp{{
  x = ::strtod(s.c_str(), nullptr);
  }}
}

function x:Integer64 <- s:String {
  cpp{{
  x = ::atol(s.c_str());
  }}
}

function x:Integer32 <- s:String {
  cpp{{
  x = ::atoi(s.c_str());
  }}
}
