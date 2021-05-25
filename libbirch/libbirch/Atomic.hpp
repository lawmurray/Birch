/**
 * @file
 */
#pragma once

/**
 * @def LIBBIRCH_ATOMIC_OPENMP
 *
 * Set to true for libbirch::Atomic to be based on std::atomic, or false to
 * use OpenMP instead.
 *
 * The advantage of the OpenMP implementation is assured memory model
 * consistency and the organic disabling of atomics when OpenMP, and thus
 * multithreading, is disabled (this can improve performance significantly for
 * single threading). The disadvantage is that OpenMP atomics do not support
 * compare-and-swap/compare-and-exchange, only swap/exchange, which requires
 * some clunkier client code, especially for read-write locks.
 *
 * The alternative implementation use std::atomic.
 *
 * Atomic provides the default constructor, copy and move constructors, copy
 * and move assignment operators, in order to be trivially copyable and so
 * a mappable type for the purposes of OpenMP. These constructors and
 * operators *do not* behave atomically, however.
 */
#ifndef _OPENMP
/* this looks like it's backwards, but when OpenMP is disabled, enabling the
 * OpenMP implementation has the effect of replacing atomic operations with
 * regular operations, which is faster */
 #define LIBBIRCH_ATOMIC_OPENMP 1
#else
/* otherwise the default is to disable the OpenMP implementation at this
 * stage, as unfortunately seeing segfaults in recent versions on macOS;
 * further review required */
#define LIBBIRCH_ATOMIC_OPENMP 1
#endif

#if !LIBBIRCH_ATOMIC_OPENMP
#include <atomic>
#endif

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
   * Default constructor. Initializes the value, not necessarily atomically.
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T value;
    #pragma omp atomic read
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
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic write
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T old;
    #pragma omp atomic capture
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
   * Apply a mask, with bitwise `and`, and return the previous value,
   * atomically.
   *
   * @param m Mask.
   *
   * @return Previous value.
   */
  T exchangeAnd(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value &= value;
    }
    return old;
    #else
    return this->value.fetch_and(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Apply a mask, with bitwise `or`, and return the previous value,
   * atomically.
   *
   * @param m Mask.
   *
   * @return Previous value.
   */
  T exchangeOr(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value |= value;
    }
    return old;
    #else
    return this->value.fetch_or(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Apply a mask, with bitwise `and`, atomically.
   *
   * @param m Mask.
   */
  void maskAnd(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
    this->value &= value;
    #else
    this->value.fetch_and(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Apply a mask, with bitwise `or`, atomically.
   *
   * @param m Mask.
   */
  void maskOr(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
    this->value |= value;
    #else
    this->value.fetch_or(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Increment the value by one, atomically, but without capturing the
   * current value.
   */
  void increment() {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
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
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
    --value;
    #else
    value.fetch_sub(1, std::memory_order_relaxed);
    #endif
  }

  /**
   * Add to the value, atomically, but without capturing the current value.
   */
  template<class U>
  void add(const U& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
    this->value += value;
    #else
    this->value.fetch_add(value, std::memory_order_relaxed);
    #endif
  }

  /**
   * Subtract from the value, atomically, but without capturing the current
   * value.
   */
  template<class U>
  void subtract(const U& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update
    this->value -= value;
    #else
    this->value.fetch_sub(value, std::memory_order_relaxed);
    #endif
  }

  template<class U>
  T operator+=(const U& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
    #if LIBBIRCH_ATOMIC_OPENMP
    T result;
    #pragma omp atomic capture
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
  #if LIBBIRCH_ATOMIC_OPENMP
  T value;
  #else
  std::atomic<T> value;
  #endif
};
}
