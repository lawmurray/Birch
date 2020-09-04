/**
 * @file
 */
#pragma once

namespace birch {
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
inline auto canonical(const birch::type::Boolean& o) {
  return o;
}

inline auto canonical(const birch::type::Real64& o) {
  return o;
}

inline auto canonical(const birch::type::Real32& o) {
  return o;
}

inline auto canonical(const birch::type::Integer64& o) {
  return o;
}

inline auto canonical(const birch::type::Integer32& o) {
  return o;
}

inline auto canonical(const birch::type::Integer16& o) {
  return o;
}

inline auto canonical(const birch::type::Integer8& o) {
  return o;
}

inline auto canonical(const birch::type::String& o) {
  return o;
}

inline auto canonical(birch::type::String&& o) {
  return std::move(o);
}

inline auto canonical(const birch::type::File& o) {
  return o;
}

}
