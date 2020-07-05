/**
 * @file
 */
#pragma once

namespace bi {
namespace type {
/*
 * Basic types.
 */
using Boolean = bool;
using Real64 = double;
using Real32 = float;
using Integer64 = std::int64_t;
using Integer32 = std::int32_t;
using Integer16 = std::int16_t;
using Integer8 = std::int8_t;
using String = std::string;
using File = FILE*;

}
}

namespace libbirch {
inline auto canonical(const bi::type::Boolean& o) {
  return o;
}

inline auto canonical(const bi::type::Real64& o) {
  return o;
}

inline auto canonical(const bi::type::Real32& o) {
  return o;
}

inline auto canonical(const bi::type::Integer64& o) {
  return o;
}

inline auto canonical(const bi::type::Integer32& o) {
  return o;
}

inline auto canonical(const bi::type::Integer16& o) {
  return o;
}

inline auto canonical(const bi::type::Integer8& o) {
  return o;
}

inline auto canonical(const bi::type::String& o) {
  return o;
}

inline auto canonical(bi::type::String&& o) {
  return std::move(o);
}

inline auto canonical(const bi::type::File& o) {
  return o;
}

}
