/**
 * @file
 */
#pragma once

namespace libbirch {
#pragma omp declare target
/**
 * Nil.
 *
 * @ingroup libbirch
 */
class Nil {
public:
  /**
   * Convert to null pointer.
   */
  operator std::nullptr_t() const {
    return nullptr;
  }

  /**
   * Convert to empty sequence.
   */
  template<class T>
  operator std::initializer_list<T>() const {
    return std::initializer_list<T>();
  }
};
#pragma omp end declare target

/**
 * Nil singleton.
 */
#pragma omp declare target
static constexpr Nil nil;
#pragma omp end declare target
}
