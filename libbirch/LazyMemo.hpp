/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Any.hpp"

namespace libbirch {
/**
 * Memo of object mappings. Keeps a memo reference to its keys, and a
 * double shared reference to its values, the latter for the purposes of
 * disabling the single-reference optimization for any objects that are
 * values in a memo. Implemented as a hash table.
 *
 * @ingroup libbirch
 */
class LazyMemo {
public:
  /**
   * Key type.
   */
  using key_type = Any*;

  /**
   * Value type.
   */
  using value_type = Any*;

  /**
   * Constructor.
   */
  LazyMemo();

  /**
   * Destructor.
   */
  ~LazyMemo();

  /**
   * Is this empty?
   */
  bool empty() const;

  /**
   * Get a value.
   *
   * @param key Key.
   *
   * @return If @p key exists, then its associated value, otherwise
   * @p failed.
   */
  value_type get(const key_type key, const value_type failed = nullptr);

  /**
   * Put an entry.
   *
   * @param key Key.
   * @param value Value.
   */
  void put(const key_type key, const value_type value);

  /**
   * Copy entries from another map into this one, removing any that are
   * obsolete.
   */
  void copy(LazyMemo& o);

  /**
   * Freeze all values in the map.
   */
  void freeze();

private:
  /**
   * Compute the hash code for a given key for a table with the given number
   * of entries.
   */
  static unsigned hash(const key_type key, const unsigned nentries);

  /**
   * Compute the lower bound on the number of occupied entries before the
   * table is considered too crowded.
   */
  unsigned crowd() const;

  /**
   * Reserve space for a new entry, rehashing the table if it has become
   * too crowded.
   */
  void reserve();
  
  /**
   * Rehash the table. This will also remove unreachable entries.
   */
  void rehash();

  /**
   * The keys.
   */
  key_type* keys;

  /**
   * The values.
   */
  value_type* values;

  /**
   * Number of entries in the table.
   */
  unsigned nentries;

  /**
   * Id of the thread that allocated keys and values.
   */
  unsigned tentries;

  /**
   * Number of occupied entries in the table.
   */
  unsigned noccupied;

  /**
   * Number of new entries since last rehash.
   */
  unsigned nnew;
};
}

inline bool libbirch::LazyMemo::empty() const {
  return nentries == 0u;
}

inline unsigned libbirch::LazyMemo::hash(const key_type key, const unsigned nentries) {
  assert(nentries > 0u);
  return static_cast<unsigned>(reinterpret_cast<size_t>(key) >> 6ull)
      & (nentries - 1u);
}

inline unsigned libbirch::LazyMemo::crowd() const {
  /* the table is considered crowded if more than three-quarters of its
   * entries are occupied */
  return (nentries >> 1u) + (nentries >> 2u);
}

#endif
