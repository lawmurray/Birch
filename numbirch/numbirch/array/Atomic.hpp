/**
 * @file
 */
#pragma once

/**
 * @internal
 * 
 * @def NUMBIRCH_ATOMIC_OPENMP
 * 
 * @ingroup array
 *
 * Set to 1 for numbirch::Atomic use OpenMP, or 0 to use std::atomic.
 *
 * The advantage of the OpenMP implementation is assured memory model
 * consistency and the organic disabling of atomics when OpenMP, and thus
 * multithreading, is disabled (this can improve performance significantly for
 * single threading). The disadvantage is that OpenMP atomics do not support
 * compare-and-swap/compare-and-exchange, only swap/exchange, which can
 * require some clunkier client code, especially for read-write locks.
 */
#define NUMBIRCH_ATOMIC_OPENMP 1

#if !NUMBIRCH_ATOMIC_OPENMP
#include <atomic>
#endif

namespace numbirch {
/**
 * @internal
 * 
 * Atomic value.
 * 
 * @ingroup array
 *
 * @tparam Value type.
 */
template<class T>
class Atomic {
public:
  /**
   * Default constructor.
   *
   * The value remains uninitialized and should be set with e.g. store(). This
   * saves an atomic write in situations where initialization is unnecessary,
   * however.
   */
  Atomic() = default;

  /**
   * Constructor.
   *
   * @param value Initial value.
   *
   * Initializes the value, atomically.
   */
  explicit Atomic(const T& value) {
    store(value);
  }

  /**
   * Load the value, atomically.
   */
  T load() const {
    #if NUMBIRCH_ATOMIC_OPENMP
    T value;
    #pragma omp atomic read relaxed
    value = this->value;
    return value;
    #else
    return this->value.load(std::memory_order_relaxed);
    #endif
  }

  /**
   * Store the value, atomically.
   */
  void store(const T& value) {
    #if NUMBIRCH_ATOMIC_OPENMP
    #pragma omp atomic write relaxed
    this->value = value;
    #else
    this->value.store(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Exchange the value with another, atomically.
   *
   * @param value New value.
   *
   * @return Old value.
   */
  T exchange(const T& value) {
    #if NUMBIRCH_ATOMIC_OPENMP
    T old;
    #pragma omp atomic capture relaxed
    {
      old = this->value;
      this->value = value;
    }
    return old;
    #else
    return this->value.exchange(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Increment the value by one, atomically, but without capturing the
   * current value.
   */
  void increment() {
    #if NUMBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update relaxed
    ++value;
    #else
    value.fetch_add(1, std::memory_order_relaxed);
    #endif
  }

  /**
   * Decrement the value by one, atomically, but without capturing the
   * current value.
   */
  void decrement() {
    #if NUMBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update relaxed
    --value;
    #else
    value.fetch_sub(1, std::memory_order_relaxed);
    #endif
  }

  template<class U>
  T operator+=(const U& value) {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      this->value += value;
      result = this->value;
    }
    return result;
    #else
    return this->value.fetch_add(value, std::memory_order_relaxed) + value;
    #endif
  }

  template<class U>
  T operator-=(const U& value) {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      this->value -= value;
      result = this->value;
    }
    return result;
    #else
    return this->value.fetch_sub(value, std::memory_order_relaxed) - value;
    #endif
  }

  T operator++() {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      ++this->value;
      result = this->value;
    }
    return result;
    #else
    return this->value.fetch_add(1, std::memory_order_relaxed) + 1;
    #endif
  }

  T operator++(int) {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      result = this->value;
      ++this->value;
    }
    return result;
    #else
    return this->value.fetch_add(1, std::memory_order_relaxed);
    #endif
  }

  T operator--() {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      --this->value;
      result = this->value;
    }
    return result;
    #else
    return this->value.fetch_sub(1, std::memory_order_relaxed) - 1;
    #endif
  }

  T operator--(int) {
    #if NUMBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture relaxed
    {
      result = this->value;
      --this->value;
    }
    return result;
    #else
    return this->value.fetch_sub(1, std::memory_order_relaxed);
    #endif
  }

private:
  /**
   * Value.
   */
  #if NUMBIRCH_ATOMIC_OPENMP
  T value;
  #else
  std::atomic<T> value;
  #endif
};
}
