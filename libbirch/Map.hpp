/**
 * @file
 */
#pragma once

#include "libbirch/Lock.hpp"

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
   * Entry type.
   */
  struct entry_type {
    /**
     * Key (source address).
     */
    std::atomic<key_type> key;

    /**
     * Value (destination address).
     */
    std::atomic<value_type> value;

    /**
     * Constructor.
     */
    entry_type() :
        key(nullptr),
        value(nullptr) {
      //
    }
  };

  /**
   * Constructor.
   *
   * @param nentries Size of the buffer. Must be a power of 2.
   */
  Map(const size_t nentries = 1 << 10);

  /**
   * Destructor.
   */
  ~Map();

  /**
   * Get a value.
   *
   * @param key The key.
   * @param fail The value on fail.
   *
   * @return If the key exists, then its value, otherwise @p fail.
   */
  value_type get(const key_type key, const value_type fail = nullptr);

  /**
   * Insert or update an entry.
   *
   * @param key The key.
   * @param value The value.
   */
  void set(const key_type key, const value_type value);

  /**
   * Insert an entry if it does not exist.
   *
   * @param key The key.
   * @param f Functional to construct value if required.
   *
   * If the key exists, its associated value is returned, otherwise a new
   * entry is inserted with a value as returned by the functional.
   *
   * @return The value.
   */
  template<class Functional>
  value_type put(const key_type key, const Functional& f);

  /**
   * Decrement the shared pointer count of all values.
   */
  void decShared();

private:
  /**
   * Find the index of an entry.
   *
   * @param key The key.
   *
   * @return If the given key exists then its index, otherwise an
   * out-of-range index.
   */
  size_t find(const key_type key) const;

  /**
   * Claim an index for an entry.
   *
   * @param key The key.
   *
   * @return If the given key exists then its index and false, otherwise an
   * empty entry is claimed and the key written, returning its index and
   * true.
   */
  std::pair<size_t,bool> claim(const key_type key);

  /**
   * Read the value at a given index.
   *
   * @param i The index.
   *
   * @return The value.
   */
  value_type read(const size_t i) const;

  /**
   * Write the value at a given index.
   *
   * @param i The index.
   * @param value The value.
   */
  void write(const size_t i, const value_type value);

  /**
   * Copy in the entries from another map.
   *
   * @param o The other map.
   *
   * This is not a thread-safe operation.
   */
  void copy(const Map& o);

  /**
   * Compute the hash for a key.
   */
  size_t hash(const key_type key) const;

  /**
   * The table.
   */
  entry_type* const entries;

  /**
   * The number of occupied entries in the table.
   */
  std::atomic<size_t> noccupied;

  /**
   * Total number of entries in the table.
   */
  const size_t nentries;

  /**
   * Resize lock.
   */
  Lock lock;
};
}

template<class Functional>
bi::Map::value_type bi::Map::put(const key_type key, const Functional& f) {
  lock.share();
  value_type result;
  auto pair = claim(key);
  if (pair.second) {
    result = f();
    write(pair.first, result);
  } else {
    result = read(pair.first);
  }
  lock.unshare();
  return result;
}
