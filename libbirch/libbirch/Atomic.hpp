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
#ifndef HAVE_OMP_H
/* this looks like it's backwards, but when OpenMP is disabled, enabling the
 * OpenMP implementation has the effect of replacing atomic operations with
 * regular operations, which is faster */
 #define LIBBIRCH_ATOMIC_OPENMP 1
#else
/* otherwise the default is to disable the OpenMP implementation at this
 * stage, as unfortunately seeing segfaults in recent versions on macOS;
 * further review required */
#define LIBBIRCH_ATOMIC_OPENMP 0
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
    T value;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic read seq_cst
    value = this->value;
    #else
    value = this->value.load();
    #endif
    return value;
  }

  /**
   * Store the value, atomically.
   */
  void store(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic write seq_cst
    this->value = value;
    #else
    this->value.store(value);
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
    T old;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    {
      old = this->value;
      this->value = value;
    }
    #else
    old = this->value.exchange(value);
    #endif
    return old;
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
    T old;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    {
      old = this->value;
      this->value &= value;
    }
    #else
    old = this->value.fetch_and(value);
    #endif
    return old;
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
    T old;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    {
      old = this->value;
      this->value |= value;
    }
    #else
    old = this->value.fetch_or(value);
    #endif
    return old;
  }

  /**
   * Apply a mask, with bitwise `and`, atomically.
   *
   * @param m Mask.
   */
  void maskAnd(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    this->value &= value;
  }

  /**
   * Apply a mask, with bitwise `or`, atomically.
   *
   * @param m Mask.
   */
  void maskOr(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    this->value |= value;
  }

  /**
   * Set to the minimum of the current value and the given value.
   * 
   * @attention The OpenMP implementation is thread-safe for multiple threads
   * calling min() simultaneously, but not for interleaving of other
   * operations, for which external synchronization is required.
   */
  void min(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    T x = value;
    T y = load();
    while (x < y) {
      y = exchange(x);
      std::swap(x, y);
    }
    #else
    T expected = std::numeric_limits<T>::max();
    while (value < expected) {
      this->value.compare_exchange_weak(expected, value);
    }
    #endif
  }

  /**
   * Set to the maximum of the current value and the given value.
   * 
   * @attention The OpenMP implementation is thread-safe for multiple threads
   * calling max() simultaneously, but not for interleaving of other
   * operations, for which external synchronization is required.
   */
  void max(const T& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    T x = value;
    T y = load();
    while (x > y) {
      y = exchange(x);
      std::swap(x, y);
    }
    #else
    T expected = std::numeric_limits<T>::min();
    while (value > expected) {
      this->value.compare_exchange_weak(expected, value);
    }
    #endif
  }

  /**
   * Increment the value by one, atomically, but without capturing the
   * current value.
   */
  void increment() {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    ++value;
  }

  /**
   * Decrement the value by one, atomically, but without capturing the
   * current value.
   */
  void decrement() {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    --value;
  }

  /**
   * Add to the value, atomically, but without capturing the current value.
   */
  template<class U>
  void add(const U& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    this->value += value;
  }

  /**
   * Subtract from the value, atomically, but without capturing the current
   * value.
   */
  template<class U>
  void subtract(const U& value) {
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic update seq_cst
    #endif
    this->value -= value;
  }

  template<class U>
  T operator+=(const U& value) {
    T result;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    result = this->value += value;
    return result;
  }

  template<class U>
  T operator-=(const U& value) {
    T result;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    result = this->value -= value;
    return result;
  }

  T operator++() {
    T value;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    value = ++this->value;
    return value;
  }

  T operator++(int) {
    T value;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    value = this->value++;
    return value;
  }

  T operator--() {
    T value;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    value = --this->value;
    return value;
  }

  T operator--(int) {
    T value;
    #if LIBBIRCH_ATOMIC_OPENMP
    #pragma omp atomic capture seq_cst
    #endif
    value = this->value--;
    return value;
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
