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
   * Start a read-only transaction.
   */
  void startRead();

  /**
   * Start a read-write transaction.
   */
  void startWrite();

  /**
   * Compute the hash code for a key.
   */
  size_t hash(const key_type key) const;

  /**
   * Get a value.
   *
   * @param key Key.
   * @param[in,out] i Index.
   *
   * @return If @p key exists, then its associated value, otherwise
   * `nullptr`.
   *
   * @p i Should be set to <tt>hash(key)</tt> or some later index where it is
   * known that @p key does not occur in the interval
   * <tt>[hash(key), i)</tt>. On exit, if @p key is found, @p i is updated to
   * its index, if @p key is not found, @p i is updated to the index of the
   * first empty entry after <tt>hash(key)</tt>.
   */
  value_type get(const key_type key, size_t& i);

  /**
   * Put a value.
   *
   * @param key Key.
   * @param value Value.
   * @param[in,out] i Index.
   *
   * @return If @p key exists, then its associated value, otherwise @p value.
   *
   * If @p key is found, works as per get(). If @p key is not found, a new
   * entry is made and associated with @p value, and @p i updated to the
   * index of this new entry.
   */
  value_type put(const key_type key, const value_type value, size_t& i);

  /**
   * Set a value.
   *
   * @param key The key.
   * @param value The value.
   * @param[in,out] i Index.
   *
   * @return @p value.
   *
   * If @p key is found, its associated value is updated to @p value and
   * @p i updated to the index of this entry. If @p key is not found a new
   * entry is made and associated with @p value and @p i updated to the index
   * of this entry.
   */
  value_type set(const key_type key, const value_type value, size_t& i);

  /**
   * Finish a read-only transaction.
   */
  void finishRead();

  /**
   * Finish a read-write transaction.
   */
  void finishWrite();

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
