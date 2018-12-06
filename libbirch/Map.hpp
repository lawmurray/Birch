/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Lock.hpp"

namespace bi {
class Memo;

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
  using key_type = Memo*;

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
   * Put a value.
   *
   * @param key Key.
   * @param value Value.
   *
   * @return If @p key exists, then its associated value, otherwise @p value.
   *
   * Use put() to not overwrite any existing value, set() to overwrite it.
   */
  value_type put(const key_type key, const value_type value);

  /**
   * Put an uninitialized value. As put(), but it is the caller's
   * responsibility to update the shared reference count on the value object
   * once it has been initialized.
   */
  value_type uninitialized_put(const key_type key, const value_type value);

  /**
   * Set a value.
   *
   * @param key The key.
   * @param value The value.
   * @param[in,out] i Index.
   *
   * @return @p value.
   *
   * Use set() to overwrite any existing value, put() to not overwrite it.
   */
  value_type set(const key_type key, const value_type value);

  /**
   * Convert shared references to values to weak references.
   */
  void weaken();

  /**
   * Destroy the map and release weak references.
   */
  void destroy();

private:
  /**
   * Joint entry type.
   */
  struct joint_entry_type {
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
   * Split entry type.
   */
  struct split_entry_type {
    /**
     * Key (source address).
     */
    std::atomic<key_type> key;

    /**
     * Value (destination address).
     */
    std::atomic<value_type> value;
  };

  /**
   * Entry type.
   */
  union entry_type {
    std::atomic<joint_entry_type> joint;
    split_entry_type split;
  };

  /**
   * Compute the hash code for a key.
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
  entry_type* entries;

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
