/**
 * @file
 */
#include "libbirch/Map.hpp"

#include <algorithm>
#include <atomic>

bi::Map::Map() :
    entries(nullptr),
    nentries(0),
    nreserved(0) {
  //
}

bi::Map::~Map() {
  deallocate(entries, nentries * sizeof(entry_type));
}

bi::Map::value_type bi::Map::get(const key_type key, const value_type fail) {
  lock.share();
  size_t i = find(key);
  auto result = (i < nentries) ? read(i) : fail;
  lock.unshare();
  return result;
}

void bi::Map::set(const key_type key, const value_type value) {
  reserve();
  lock.share();
  auto result = claim(key);
  write(result.first, value);
  if (!result.second) {
    unreserve();
  }
  lock.unshare();
}

void bi::Map::decShared() {
  key_type key, value;
  for (size_t i = 0; i < nentries; ++i) {
    key = entries[i].key.load();
    if (key) {
      value = entries[i].value.load();
      if (value) {
        value->decShared();
      }
    }
  }
}

size_t bi::Map::find(const key_type key) const {
  if (nentries > 0) {
    size_t i = hash(key);
    key_type k = entries[i].key.load();
    while (k && k != key) {
      i = (i + 1) & (nentries - 1);
      k = entries[i].key.load();
    }
    return (k == key) ? i : nentries;
  } else {
    return 0ull;
  }
}

std::pair<size_t,bool> bi::Map::claim(const key_type key) {
  size_t i = hash(key);
  key_type k = nullptr;
  entries[i].key.compare_exchange_strong(k, key);
  while (k && k != key) {
    k = nullptr;
    i = (i + 1) & (nentries - 1);
    entries[i].key.compare_exchange_strong(k, key);
  }
  return std::make_pair(i, k == nullptr);
}

bi::Map::value_type bi::Map::read(const size_t i) const {
  assert(i < nentries);
  value_type value;
  do {
    /* must spin here as the entry may have been reserved, with the write of
     * the value pending */
    value = entries[i].value.load();
  } while (!value);
  return value;
}

void bi::Map::write(const size_t i, const value_type value) {
  assert(i < nentries);
  return entries[i].value.store(value);
}

size_t bi::Map::hash(const key_type key) const {
  assert(nentries > 0);
  return (reinterpret_cast<size_t>(key) >> 6ull) & (nentries - 1ull);
}

void bi::Map::reserve() {
  /* the table is considered crowded if more than three-quarters of its
   * entries are occupied */
  if (++nreserved > (nentries >> 1ull) + (nentries >> 2ull)) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (nreserved > (nentries >> 1ull) + (nentries >> 2ull)) {
      /* double size */
      size_t nentries1 = std::max(2ull*nentries, 256ull);

      /* keep doubling until no longer crowded (other threads may be
       * reserving in the meantime */
      while (nreserved > (nentries1 >> 1ull) + (nentries1 >> 2ull)) {
        nentries1 <<= 1ull;
      }

      /* resize */
      entry_type* entries1 = (entry_type*)allocate(nentries1*sizeof(entry_type));
      std::memset(entries1, 0, nentries1*sizeof(entry_type));
      std::swap(nentries1, nentries);
      std::swap(entries1, entries);

      /* copy over previous entries */
      for (size_t i = 0; i < nentries1; ++i) {
        key_type key = entries1[i].key.load();
        if (key) {
          set(key, entries1[i].value.load());
        }
      }

      /* deallocate previous */
      deallocate(entries1, nentries1*sizeof(entry_type));
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Map::unreserve() {
  --nreserved;
}
