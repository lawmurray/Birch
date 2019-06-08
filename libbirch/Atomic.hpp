/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Atomic value.
 *
 * @tparam Value type.
 */
template<class T>
class Atomic {
public:
  /**
   * Constructor.
   *
   * @param value Initial value.
   */
  Atomic(const T& value) {
    #pragma omp atomic write
    this->value = value;
  }

  /**
   * Copy constructor.
   */
  Atomic(const Atomic<T>& o) {
    #pragma omp atomic write
    this->value = o.load();
  }

  /**
   * Value assignment operator.
   */
  Atomic<T>& operator=(const Atomic<T>& o) {
    store(o.load());
    return *this;
  }

  /**
   * Assignment operator.
   */
  Atomic<T>& operator=(const T& value) {
    store(value);
    return *this;
  }

  /**
   * Load the value, atomically.
   */
  T load() const {
    T value;
    #pragma omp atomic read
    value = this->value;

    return value;
  }

  /**
   * Store the value, atomically.
   */
  void store(const T& value) {
    #pragma omp atomic write
    this->value = value;
  }

  /**
   * Exchange the value with another, atomically.
   *
   * @param value New value.
   *
   * @return Old value.
   */
  T exchange(const T& value) {
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value = value;
    }
    return old;
  }

private:
  /**
   * Value.
   */
  T value;
};
}
