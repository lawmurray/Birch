/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
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
   * Constructor.
   */
  Map();

  /**
   * Destructor.
   */
  ~Map();

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
   *
   * @return If @p key exists, then its associated value, otherwise @p value.
   */
  value_type put(const key_type key, const value_type value);

  /**
   * Put an uninitialized value. As put(), but it is the caller's
   * responsibility to update reference counts on the key (weak) and value
   * (shared).
   *
   * @param key Key.
   * @param value Value.
   *
   * @return If @p key exists, then its associated value, otherwise @p value.
   */
  value_type uninitialized_put(const key_type key, const value_type value);

  /**
   * Remove an entry.
   *
   * @param key Key.
   * @param value Value.
   *
   * An entry is removed by overwriting its key with a bit pattern that
   * indicates erased. Consequently, the entry still occupies a slot. If the
   * map is later resized, these erased entries will be removed.
   */
  void remove(const key_type key);

  /**
   * Erase all entries in the map that are unreachable.
   */
  void clean();

  /**
   * Freeze all values in the map.
   */
  void freeze();

private:
  /**
   * Get a value.
   *
   * @param i Index.
   *
   * @return The value.
   */
  value_type get(const unsigned i);

  /**
   * Compute the hash code for a key.
   */
  unsigned hash(const key_type key) const;

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
   * The keys.
   */
  std::atomic<key_type>* keys;

  /**
   * The values.
   */
  std::atomic<value_type>* values;

  /**
   * Number of entries in the table.
   */
  unsigned nentries;

  /**
   * Number of occupied entries in the table.
   */
  std::atomic<unsigned> noccupied;

  /**
   * Resize lock.
   */
  Lock lock;
};
}
