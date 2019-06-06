/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Atomic value. Adds a copy constructor to std::atomic in order to create
 * a mappable type for OpenMP copies between host and device.
 */
template<class T>
class Atomic : public std::atomic<T> {
public:
  Atomic() = default;

  Atomic(const T& value) : std::atomic<T>(value) {
    //
  }

  Atomic(const Atomic<T>& o) : std::atomic<T>(o.load()) {
    //
  }
};
}
