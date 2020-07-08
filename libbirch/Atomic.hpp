/**
 * @file
 */
#pragma once

//#include <atomic>

namespace libbirch {
/**
 * Atomic value.
 *
 * @tparam Value type.
 *
 * The implementation uses OpenMP atomics as opposed to std::atomic. The
 * advantage of this is ensured memory model consistency and the organic
 * disabling of atomics when OpenMP, and thus multithreading, is
 * disabled (this can improve performance significantly for single threading).
 * The disadvantage is that OpenMP atomics do not support
 * compare-and-swap/compare-and-exchange, only swap/exchange, which requires
 * some clunkier client code, especially for read-write locks.
 *
 * Atomic provides the default constructor, copy and move constructors, copy
 * and move assignment operators, in order to be trivially copyable and so
 * a mappable type for the purposes of OpenMP. These constructors and
 * operators *do not* behave atomically, however.
 *
 * @internal An alternative implementation, in comments, uses std::atomic.
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
   * @param value Initial value. Initializes the value, not necessarily
   * atomically.
   */
  explicit Atomic(const T& value) :
      value(value) {
    //
  }

  /**
   * Get the value, not necessarily atomically.
   */
  T get() const {
    //return value.load();
    return value;
  }

  /**
   * Get the value, not necessarily atomically.
   */
  void set(const T& value) {
    //return this->value.store(value);
    this->value = value;
  }

  /**
   * Load the value, atomically.
   */
  T load() const {
    //return this->value.load();
    T value;
    #pragma omp atomic read
    value = this->value;
    return value;
  }

  /**
   * Store the value, atomically.
   */
  void store(const T& value) {
    //return this->value.store(value);
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
    //return this->value.exchange(value);
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value = value;
    }
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
    //return this->value.fetch_and(value);
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value &= value;
    }
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
    //return this->value.fetch_or(value);
    T old;
    #pragma omp atomic capture
    {
      old = this->value;
      this->value |= value;
    }
    return old;
  }

  /**
   * Apply a mask, with bitwise `and`, atomically.
   *
   * @param m Mask.
   */
  void maskAnd(const T& value) {
    //this->value &= value;
    #pragma omp atomic update
    this->value &= value;
  }

  /**
   * Apply a mask, with bitwise `or`, atomically.
   *
   * @param m Mask.
   */
  void maskOr(const T& value) {
    //this->value |= value;
    #pragma omp atomic update
    this->value |= value;
  }

  /**
   * Increment the value by one, atomically, but without capturing the
   * current value.
   */
  void increment() {
    //++value;
    #pragma omp atomic update
    ++value;
  }

  /**
   * Decrement the value by one, atomically, but without capturing the
   * current value.
   */
  void decrement() {
    //--value;
    #pragma omp atomic update
    --value;
  }

  /**
   * Add to the value, atomically, but without capturing the current value.
   */
  template<class U>
  void add(const U& value) {
    //this->value += value;
    #pragma omp atomic update
    this->value += value;
  }

  /**
   * Subtract from the value, atomically, but without capturing the current
   * value.
   */
  template<class U>
  void subtract(const U& value) {
    //this->value -= value;
    #pragma omp atomic update
    this->value -= value;
  }

  template<class U>
  T operator+=(const U& value) {
    //return this->value += value;
    T result;
    #pragma omp atomic capture
    result = this->value += value;
    return result;
  }

  template<class U>
  T operator-=(const U& value) {
    //return this->value -= value;
    T result;
    #pragma omp atomic capture
    result = this->value -= value;
    return result;
  }

  T operator++() {
    //return ++this->value;
    T value;
    #pragma omp atomic capture
    value = ++this->value;
    return value;
  }

  T operator++(int) {
    //return this->value++;
    T value;
    #pragma omp atomic capture
    value = this->value++;
    return value;
  }

  T operator--() {
    //return --this->value;
    T value;
    #pragma omp atomic capture
    value = --this->value;
    return value;
  }

  T operator--(int) {
    //return this->value--;
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
  //std::atomic<T> value;
};
}
