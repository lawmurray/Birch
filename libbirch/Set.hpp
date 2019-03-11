/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"
#include "libbirch/Lock.hpp"

#include <atomic>

namespace libbirch {
/**
 * Thread-safe set of memos.
 *
 * @ingroup libbirch
 *
 * The implementation is lock-free except when resizing is required.
 */
class Set {
public:
  /**
   * Value type.
   */
  using value_type = Memo*;

  /**
   * Constructor.
   */
  Set();

  /**
   * Destructor.
   */
  ~Set();

  /**
   * Is this empty?
   */
  bool empty() const;

  /**
   * Does the set contain a value?
   */
  bool contains(const value_type value);

  /**
   * Insert a value into the set. If the value is already in the set, does
   * nothing.
   */
  void insert(const value_type value);

private:
  /**
   * Compute the hash code for a value.
   */
  unsigned hash(const value_type value) const;

  /**
   * Compute the lower bound on reserved entries to be considered crowded.
   */
  unsigned crowd() const;

  /**
   * Reserve space for a (possible) new entry, resizing if necessary.
   */
  void reserve();

  /**
   * Release a reservation previously obtained with reserve(), which will
   * not be needed.
   */
  void unreserve();

  /**
   * The values.
   */
  std::atomic<value_type>* values;

  /**
   * Number of entries.
   */
  unsigned nentries;

  /**
   * Id of the thread that allocated values.
   */
  unsigned tentries;

  /**
   * Number of occupied entries.
   */
  std::atomic<unsigned> noccupied;

  /**
   * Resize lock.
   */
  Lock lock;
};
}

inline bool libbirch::Set::empty() const {
  return nentries == 0u;
}

inline unsigned libbirch::Set::hash(const value_type value) const {
  assert(nentries > 0u);
  return (reinterpret_cast<size_t>(value) >> 5ull) & (nentries - 1u);
}

inline unsigned libbirch::Set::crowd() const {
  /* the set is considered crowded if more than three-quarters of its
   * entries are occupied */
  return (nentries >> 1u) + (nentries >> 2u);
}

inline void libbirch::Set::unreserve() {
  noccupied.fetch_sub(1u, std::memory_order_relaxed);
}
