/**
 * @file
 */
#include "libbirch/Map.hpp"

#include <algorithm>

bi::Map::Map(const unsigned mn) :
    entries(nullptr),
    nentries(0),
    nreserved(0),
    mn(mn) {
  //
}

bi::Map::~Map() {
  deallocate(entries, nentries * sizeof(std::atomic<entry_type>));
}

bi::Map::value_type bi::Map::get(const key_type key, const value_type fail) {
  assert(key);
  value_type value = fail;
  if (nentries > 0) {
    lock.share();
    size_t i = hash(key);
    entry_type entry = entries[i].load(std::memory_order_relaxed);
    while (entry.key && entry.key != key) {
      i = (i + 1) & (nentries - 1);
      entry = entries[i].load(std::memory_order_relaxed);
    }
    lock.unshare();
    if (entry.key == key) {
      value = entry.value;
    }
  }
  return value;
}

void bi::Map::put(const key_type key, const value_type value) {
  assert(key);
  assert(value);
  reserve();
  lock.share();
  size_t i = hash(key);
  entry_type expected = { nullptr, nullptr };
  entry_type desired = { key, value };
  while (!entries[i].compare_exchange_strong(expected, desired, std::memory_order_relaxed)) {
    i = (i + 1) & (nentries - 1);
    expected = {nullptr, nullptr};
  }
  lock.unshare();
}

void bi::Map::decShared() {
  entry_type* entries1 = (entry_type*)entries;
  for (size_t i = 0; i < nentries; ++i) {
    entry_type entry = entries1[i];
    if (entry.key) {
      entry.value->decShared();
    }
  }
}

size_t bi::Map::hash(const key_type key) const {
  assert(nentries > 0);
  return (reinterpret_cast<size_t>(key) >> 6ull) & (nentries - 1ull);
}

size_t bi::Map::crowd() const {
  /* the table is considered crowded if more than three-quarters of its
   * entries are occupied */
  return (nentries >> 1ull) + (nentries >> 2ull);
}

void bi::Map::reserve() {
  size_t nreserved1 = ++nreserved;
  if (nreserved1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (nreserved1 > crowd()) {
      /* save previous table */
      size_t nentries1 = nentries;
      entry_type* entries1 = (entry_type*)entries;

      /* initialize new table */
      size_t nentries2 = std::max(2ull * nentries1, (unsigned long long)mn);
      entry_type* entries2 = (entry_type*)allocate(nentries2 * sizeof(entry_type));
      std::memset(entries2, 0, nentries2 * sizeof(entry_type));

      /* copy contents from previous table */
      nentries = nentries2;
      for (size_t i = 0u; i < nentries1; ++i) {
        entry_type entry = entries1[i];
        if (entry.key) {
          size_t j = hash(entry.key);
          while (entries2[j].key) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          entries2[j] = entry;
        }
      }
      entries = (std::atomic<entry_type>*)entries2;

      /* deallocate previous table */
      deallocate(entries1, nentries1 * sizeof(entry_type));
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Map::unreserve() {
  --nreserved;
}
