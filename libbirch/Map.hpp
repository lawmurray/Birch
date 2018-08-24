/**
 * @file
 */
#pragma once

namespace bi {
class Any;
/**
 * Thread-safe lock-free hash table of memory mappings.
 *
 * @ingroup libbirch
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
    entry_type() : key(nullptr), value(nullptr) {
      //
    }
  };

  /**
   * Constructor.
   *
   * @param nentries Size of the buffer. Must be a power of 2.
   */
  Map(const size_t nentries = 1 << 16);

  /**
   * Destructor.
   */
  ~Map();

  /**
   * @name High-level interface.
   */
  //@{
  /**
   * Get the value for a given key.
   *
   * @param key The key,
   *
   * @return The value.
   */
  value_type get(const key_type key) const;

  /**
   * Set the value for a given index.
   *
   * @param key The key,
   * @param value The value.
   *
   * This will fail in an infinite loop if the hash table is full. To ensure
   * that this does not occur, call reserve() before inserting each new key.
   */
  void set(const key_type key, const value_type value);

  /**
   * Reserve an entry.
   *
   * @return True if there is space for at least one more insertion, false
   * otherwise.
   */
  bool reserve();
  //@}

  /**
   * @name Low-level interface.
   */
  //@{
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
   * Decrement the shared pointer count of all values.
   */
  void decShared();
  //@}

private:
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
};
}
