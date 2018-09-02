/**
 * @file
 */
#include "libbirch/Map.hpp"

#include <algorithm>
#include <functional>

bi::Map::Map(const unsigned mn) :
    entries(nullptr),
    nentries(0),
    nreserved(0),
    mn(mn) {
  //
}

bi::Map::~Map() {
  joint_entry_type* entries1 = (joint_entry_type*)entries;
  for (size_t i = 0; i < nentries; ++i) {
    joint_entry_type entry = entries1[i];
    if (entry.key) {
      entry.value->decShared();
    }
  }
  deallocate(entries, nentries * sizeof(entry_type));
}

bi::Map::value_type bi::Map::get(const key_type key, const value_type fail) {
  assert(key);

  size_t i;
  bool found;
  value_type value = fail;

  if (nentries > 0) {
    lock.share();
    std::tie(i, found) = find(key, hash(key));
    if (found) {
      value = entries[i].split.value.load(std::memory_order_relaxed);
    }
    lock.unshare();
  }
  return value;
}

void bi::Map::set(const key_type key, const value_type value) {
  assert(key);
  assert(value);
  assert(nentries > 0);

  size_t i;
  bool found;

  lock.share();
  std::tie(i, found) = find(key, hash(key));
  assert(found);
  value_type old = entries[i].split.value.load(std::memory_order_relaxed);
  entries[i].split.value.store(value, std::memory_order_relaxed);
  lock.unshare();
  value->incShared();
  old->decShared();
}

void bi::Map::put(const key_type key, const value_type value) {
  reserve();
  lock.share();
  insert(key, value, hash(key));
  lock.unshare();
  value->incShared();
}

bi::Map::value_type bi::Map::getOrPut(const key_type key,
    const std::function<value_type()>& f) {
  assert(key);

  size_t i;
  bool found;
  value_type value;

  reserve();
  lock.share();
  std::tie(i, found) = find(key, hash(key));
  if (found) {
    unreserve();  // key exists, cancel reservation for insert
    value = entries[i].split.value.load(std::memory_order_relaxed);
  } else {
    value = f();
    insert(key, value, i);
    value->incShared();
  }

  lock.unshare();
  return value;
}

void bi::Map::setOrPut(const key_type key, const value_type value) {
  assert(key);
  assert(value);

  size_t i;
  bool found;

  reserve();
  lock.share();
  std::tie(i, found) = find(key, hash(key));
  if (found) {
    unreserve();  // key exists, cancel reservation for insert
    value_type old = entries[i].split.value.load(std::memory_order_relaxed);
    entries[i].split.value.store(value, std::memory_order_relaxed);
    value->incShared();
    old->decShared();
  } else {
    insert(key, value, i);
    value->incShared();
  }
  lock.unshare();
}

std::pair<size_t,bool> bi::Map::find(const key_type key, const size_t start) {
  size_t i = start;
  key_type k = entries[i].split.key.load(std::memory_order_relaxed);
  while (k && k != key) {
    i = (i + 1) & (nentries - 1);
    k = entries[i].split.key.load(std::memory_order_relaxed);
  }
  return std::make_pair(i, k == key);
}

void bi::Map::insert(const key_type key, const value_type value,
    const size_t start) {
  size_t i = start;
  joint_entry_type expected = { nullptr, nullptr };
  joint_entry_type desired = { key, value };
  while (!entries[i].joint.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected.key != key) {
    i = (i + 1) & (nentries - 1);
    expected = {nullptr, nullptr};
  }
  if (expected.key == key) {
    entries[i].split.value = value;
  }
}

size_t bi::Map::hash(const key_type key) const {
  assert(nentries > 0);
  return (reinterpret_cast<size_t>(key) >> 5ull) & (nentries - 1ull);
}

size_t bi::Map::crowd() const {
  /* the table is considered crowded if more than three-quarters of its
   * entries are occupied */
  return (nentries >> 1ull) + (nentries >> 2ull);
}

void bi::Map::reserve() {
  size_t nreserved1 = nreserved.fetch_add(1u) + 1u;
  if (nreserved1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (nreserved1 > crowd()) {
      /* save previous table */
      size_t nentries1 = nentries;
      joint_entry_type* entries1 = (joint_entry_type*)entries;

      /* initialize new table */
      size_t nentries2 = std::max(2ull * nentries1, (unsigned long long)mn);
      joint_entry_type* entries2 = (joint_entry_type*)allocate(
          nentries2 * sizeof(entry_type));
      std::memset(entries2, 0, nentries2 * sizeof(entry_type));

      /* copy contents from previous table */
      nentries = nentries2;
      for (size_t i = 0u; i < nentries1; ++i) {
        joint_entry_type entry = entries1[i];
        if (entry.key) {
          size_t j = hash(entry.key);
          while (entries2[j].key) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          entries2[j] = entry;
        }
      }
      entries = (entry_type*)entries2;

      /* deallocate previous table */
      deallocate(entries1, nentries1 * sizeof(joint_entry_type));
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Map::unreserve() {
  nreserved.fetch_sub(1u, std::memory_order_relaxed);
}
