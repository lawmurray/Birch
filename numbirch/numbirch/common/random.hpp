/**
 * @file
 */
#pragma once

#include <random>

namespace numbirch {
/**
 * @internal
 * 
 * 32-bit pseudorandom number generator for each host thread.
 */
extern thread_local std::mt19937 rng32;

/**
 * @internal
 * 
 * 64-bit pseudorandom number generator for each host thread.
 */
extern thread_local std::mt19937_64 rng64;

/**
 * @internal
 * 
 * Templated access to required functions and objects for single and double
 * precision.
 */
template<class T>
struct stl {
  //
};
template<>
struct stl<double> {
  static auto& rng() {
    return rng64;
  }
};
template<>
struct stl<float> {
  static auto& rng() {
    return rng32;
  }
};
template<>
struct stl<int> {
  static auto& rng() {
    return rng32;
  }
};
template<>
struct stl<bool> {
  static auto& rng() {
    return rng32;
  }
};

}
