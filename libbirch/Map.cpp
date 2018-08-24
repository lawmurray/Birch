/**
 * @file
 */
#include "libbirch/Map.hpp"

#include <atomic>

bi::Map::Map(const size_t nentries) :
    entries((entry_type*)bi::allocate(nentries * sizeof(entry_type))),
    noccupied(0),
    nentries(nentries) {
  std::memset(entries, 0, nentries * sizeof(entry_type));
}

bi::Map::~Map() {
  bi::deallocate(entries, nentries * sizeof(entry_type));
}

bi::Map::value_type bi::Map::get(const key_type key) const {
  size_t i = find(key);
  return (i < nentries) ? read(i) : nullptr;
}

void bi::Map::set(const key_type key, const value_type value) {
  auto result = claim(key);
  write(result.first, value);
}

bool bi::Map::reserve() {
  /* the table is considered full if more than three-quarters of its
   * entries are occupied */
  return ++noccupied <= (nentries >> 1) + (nentries >> 2);
}

size_t bi::Map::find(const key_type key) const {
  size_t i = hash(key);
  key_type k = entries[i].key.load();
  while (k && k != key) {
    i = (i + 1) & (nentries - 1);
    k = entries[i].key.load();
  }
  return (k == key) ? i : nentries;
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
  value_type value;
  do {
    /* must spin here as the entry may have been reserved, with the write of
     * the value pending */
    value = entries[i].value.load();
  } while (!value);
  return value;
}

void bi::Map::write(const size_t i, const value_type value) {
  return entries[i].value.store(value);
}

void bi::Map::copy(const Map& o) {
  key_type key;
  for (size_t i = 0; i < o.nentries; ++i) {
    key = o.entries[i].key.load();
    if (key) {
      set(key, o.entries[i].value.load());
    }
  }
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

size_t bi::Map::hash(const key_type key) const {
  return (reinterpret_cast<size_t>(key) >> 6) & (nentries - 1);
}
