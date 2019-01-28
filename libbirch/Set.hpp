/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Lock.hpp"

namespace bi {
class Memo;

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
   * Entry type.
   */
  using entry_type = std::atomic<value_type>;

  /**
   * Compute the hash code for a value.
   */
  size_t hash(const value_type value) const;

  /**
   * Compute the lower bound on reserved entries to be considered crowded.
   */
  size_t crowd() const;

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
   * Entries.
   */
  entry_type* entries;

  /**
   * Number of entries.
   */
  size_t nentries;

  /**
   * Number of occupied entries.
   */
  std::atomic<size_t> noccupied;

  /**
   * Resize lock.
   */
  Lock lock;
};
}
