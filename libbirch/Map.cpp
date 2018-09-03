/**
 * @file
 */
#include "libbirch/Map.hpp"

bi::Map::Map() :
    entries(nullptr),
    nentries(0),
    nreserved(0) {
  //
}

bi::Map::~Map() {
  joint_entry_type* entries1 = (joint_entry_type*)entries;
  for (size_t i = 0; i < nentries; ++i) {
    joint_entry_type entry = entries1[i];
    if (entry.key) {
      //entry.key->decWeak();
      entry.value->decShared();
    }
  }
  deallocate(entries, nentries * sizeof(entry_type));
}

bool bi::Map::empty() const {
  return nentries == 0;
}

bi::Map::value_type bi::Map::get(const key_type key, const value_type failed) {
  /* pre-condition */
  assert(key);

  value_type result = failed;
  if (!empty()) {
    lock.share();
    size_t i = hash(key);

    key_type k = entries[i].split.key.load(std::memory_order_relaxed);
    while (k && k != key) {
      i = (i + 1) & (nentries - 1);
      k = entries[i].split.key.load(std::memory_order_relaxed);
    }
    if (k == key) {
      result = entries[i].split.value.load(std::memory_order_relaxed);
    }
    lock.unshare();
  }
  return result;
}

bi::Map::value_type bi::Map::put(const key_type key, const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  //key->incWeak();
  value->incShared();
  reserve();
  lock.share();

  joint_entry_type expected = { nullptr, nullptr };
  joint_entry_type desired = { key, value };

  size_t i = hash(key);
  while (!entries[i].joint.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected.key != key) {
    i = (i + 1) & (nentries - 1);
    expected = {nullptr, nullptr};
  }

  value_type result;
  if (expected.key == key) {
    unreserve();  // key exists, cancel reservation for insert
    result = expected.value;
    //key->decWeak();
    value->decShared();
  } else {
    result = value;
  }
  lock.unshare();
  return result;
}

bi::Map::value_type bi::Map::set(const key_type key, const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  //key->incWeak();
  value->incShared();
  reserve();
  lock.share();

  joint_entry_type expected = { nullptr, nullptr };
  joint_entry_type desired = { key, value };

  size_t i = hash(key);
  while (!entries[i].joint.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected.key != key) {
    i = (i + 1) & (nentries - 1);
    expected = {nullptr, nullptr};
  }

  if (expected.key == key) {
    unreserve();  // key exists, cancel reservation for insert
    value_type old = expected.value;
    while (!entries[i].split.value.compare_exchange_weak(old, value,
        std::memory_order_relaxed));
    //key->decWeak();
    old->decShared();
  }
  lock.unshare();
  return value;
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
      size_t nentries2 = std::max(2ull * nentries1, 64ull);
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
