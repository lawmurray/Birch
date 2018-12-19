/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Lock.hpp"

namespace bi {
class Memo;

/**
 * Thread-safe list of memos.
 *
 * @ingroup libbirch
 *
 * The implementation is lock-free except when resizing is required.
 */
class List {
public:
  /**
   * Value type.
   */
  using value_type = Memo*;

  /**
   * Constructor.
   */
  List();

  /**
   * Destructor.
   */
  ~List();

  /**
   * Is this empty?
   */
  bool empty() const;

  /**
   * Put a value.
   *
   * @param value Value.
   */
  void put(const value_type value);

  /**
   * Remove an entry from each memo in the list, and destroy the list.
   *
   * @param key Key of the entry to remove from each memo.
   */
  void destroy(Any* key);

public:
  /**
   * Reserve space for a new entry, resizing if necessary.
   *
   * @return Index of the reserved space.
   */
  size_t reserve();

  /**
   * The table.
   */
  value_type* entries;

  /**
   * Number of entries in the table.
   */
  size_t nentries;

  /**
   * Number of occupied entries in the table.
   */
  std::atomic<size_t> noccupied;

  /**
   * Resize lock.
   */
  Lock lock;
};
}
