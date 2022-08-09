/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"

namespace membirch {
class Any;

/*
 * Memo for copying graphs of unknown size, implemented as a hash map that is
 * resized and rehashed as needed.
 */
class Memo {
public:
  /**
   * Constructor.
   */
  Memo();

  /**
   * Destructor.
   */
  ~Memo();

  /**
   * Get reference to the value associated with a key, which may be `nullptr`,
   * in which case the value may be written.
   *
   * @param key Key.
   */
  Any*& get(Any* key);

private:
  /**
   * Compute the hash code for a given key.
   */
  int hash(Any* key) const;

  /**
   * Compute the lower bound on the number of occupied entries before the
   * table is considered too crowded.
   */
  int crowd() const;

  /**
   * Rehash the table.
   */
  void rehash();

  /**
   * The keys.
   */
  Any** keys;

  /**
   * The values.
   */
  Any** values;

  /**
   * Number of entries in the table.
   */
  int nentries;

  /**
   * Number of occupied entries in the table.
   */
  int noccupied;

  /**
   * Size of a newly-allocated table. As #keys and #values are allocated
   * separately as arrays of pointers, an initial size of 8 is 64 bytes each,
   * a common cache line size.
   */
  static constexpr int INITIAL_SIZE = 8;
};
}
