/**
 * @file
 */
#pragma once

#include "libbirch/Lock.hpp"

#include <omp.h>

namespace bi {
class Any;
/**
 * Thread-safe hash table of memory mappings.
 *
 * @ingroup libbirch
 *
 * The implementation is lock-free except when resizing is required.
 */
class Map {
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
  Map();

  /**
   * Destructor.
   */
  ~Map();

  /**
   * Get a value by key.
   *
   * @param key The key.
   * @param fail The value on fail.
   *
   * @return If the key exists, then its value, otherwise @p fail.
   */
  value_type get(const key_type key, const value_type fail = nullptr);

  /**
   * Insert a value by key, assuming that the key does not already exist.
   *
   * @param key The key.
   * @param value The value.
   *
   * If the key exists, its associated value is returned, otherwise a new
   * entry is inserted with a value as returned by the functional.
   *
   * @return The value.
   */
  void put(const key_type key, const value_type value);

  /**
   * Get a value by key, or insert it if it doesn't exist.
   *
   * @param key The key.
   * @param f Functional to produce the value if insertion is required.
   *
   * If the key exists, its associated value is returned, otherwise a value
   * is generated from the functional and inserted.
   *
   * @return The value.
   */
  template<class Functional>
  value_type getOrPut(const key_type key, const Functional& f);

  /**
   * Decrement the shared pointer count of all values.
   */
  void decShared();

private:
  /**
   * Entry type.
   */
  struct entry_type {
    /**
     * Key (source address).
     */
    key_type key;

    /**
     * Value (destination address).
     */
    value_type value;
  };

  /**
   * Compute the hash for a key.
   */
  size_t hash(const key_type key) const;

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
   * The table.
   */
  std::atomic<entry_type>* entries;

  /**
   * Total number of entries in the table.
   */
  size_t nentries;

  /**
   * Number of reserved entries in the table.
   */
  std::atomic<size_t> nreserved;

  /**
   * Resize lock.
   */
  Lock lock;
};
}

template<class Functional>
bi::Map::value_type bi::Map::getOrPut(const key_type key,
    const Functional& f) {
  assert(key);
  reserve();
  lock.share();

  /* try get */
  size_t i = hash(key);
  entry_type entry = entries[i].load();
  while (entry.key && entry.key != key) {
    i = (i + 1u) & (nentries - 1u);
    entry = entries[i].load();
  }
  if (entry.key == key) {
    /* get succeeded, cancel reservation */
    unreserve();
  } else {
    /* get failed, do put instead */
    entry = {key, f()};
    entry_type expected = {nullptr, nullptr};
    while (!entries[i].compare_exchange_strong(expected, entry)) {
      i = (i + 1u) & (nentries - 1u);
      expected = {nullptr, nullptr};
    }
  }

  lock.unshare();
  return entry.value;
}
