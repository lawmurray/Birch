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
   *
   * @param mn Minimum size of table.
   */
  Map(const unsigned mn = 256u);

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
   * Set a value by key, assuming that the key already exists.
   *
   * @param key The key.
   * @param value The value.
   */
  void set(const key_type key, const value_type value);

  /**
   * Set a value by key, assuming that the key does not already exist.
   *
   * @param key The key.
   * @param value The value.
   */
  void put(const key_type key, const value_type value);

  /**
   * Get a value by key, or set it if it doesn't exist.
   *
   * @param key The key.
   * @param f Function to produce the value if insertion is required.
   *
   * If the key exists, its associated value is returned, otherwise a value
   * is generated from the functional and inserted.
   *
   * @return The value.
   */
  value_type getOrPut(const key_type key,
      const std::function<value_type()>& f);

  /**
   * Set a value by key, or put it if it doesn't exist.
   *
   * @param key The key.
   * @param value The value.
   */
  void setOrPut(const key_type key, const value_type value);

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
   * Find by key.
   *
   * @param key The key.
   * @param start Starting index for the search.
   *
   * @return If the key exists, then its index and true, otherwise the index
   * of the first empty entry where the key could be inserted and false.
   */
  std::pair<size_t,bool> find(const key_type key, const size_t start);

  /**
   * Insert by key.
   *
   * @param key The key.
   * @param value The value.
   * @param start Starting index for the search.
   */
  void insert(const key_type key, const value_type value, const size_t start);

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

  /**
   * Minimum size of table.
   */
  unsigned mn;
};
}
