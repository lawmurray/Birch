/**
 * @file
 */
#pragma once

namespace libbirch {
#pragma omp declare target
/**
 * Atomic value.
 *
 * @tparam Value type.
 *
 * The implementation uses OpenMP atomics as opposed to std::atomic. The
 * advantage of this is ensured memory model consistency and the organic
 * disabling of atomics when OpenMP, and thus multithreading, is
 * disabled. The disadvantage is that OpenMP atomics do not support
 * compare-and-swap/compare-and-exchange, only swap/exchange, which requires
 * some clunkier client code, especially for read-write locks.
 */
template<class T>
class Atomic {
public:
  /**
   * Constructor.
   *
   * @param value Initial value.
   */
  explicit Atomic(const T& value) {
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
   * Assignment operator.
   */
  Atomic<T>& operator=(const Atomic<T>& o) {
    store(o.load());
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

  template<class U>
  T operator+=(const U& value) {
    T result;
    #pragma omp atomic capture
    result = this->value += value;

    return result;
  }

  template<class U>
  T operator-=(const U& value) {
    T result;
    #pragma omp atomic capture
    result = this->value -= value;

    return result;
  }

  T operator++() {
    T value;
    #pragma omp atomic capture
    value = ++this->value;

    return value;
  }

  T operator++(int) {
    T value;
    #pragma omp atomic capture
    value = this->value++;

    return value;
  }

  T operator--() {
    T value;
    #pragma omp atomic capture
    value = --this->value;

    return value;
  }

  T operator--(int) {
    T value;
    #pragma omp atomic capture
    value = this->value--;

    return value;
  }

private:
  /**
   * Value.
   */
  T value;
};
#pragma omp end declare target
}
