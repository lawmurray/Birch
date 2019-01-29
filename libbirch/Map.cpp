/**
 * @file
 */
#include "libbirch/Map.hpp"

static bi::Any* const EMPTY = nullptr;
static bi::Any* const ERASED = reinterpret_cast<bi::Any*>(0xFFFFFFFFFFFFFFFF);

bi::Map::Map() :
    entries(nullptr),
    nentries(0),
    noccupied(0) {
  //
}

bi::Map::~Map() {
  for (size_t i = 0; i < nentries; ++i) {
    joint_entry_type entry = entries[i].joint.load(std::memory_order_relaxed);
    if (entry.key != EMPTY && entry.key != ERASED) {
      entry.key->decMemo();
      entry.value->decShared();
    }
  }
  deallocate(entries, nentries * sizeof(entry_type));
}

bool bi::Map::empty() const {
  return nentries == 0;
}

bi::Map::value_type bi::Map::get(const key_type key,
    const value_type failed) {
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

  key->incMemo();
  value->incShared();

  reserve();
  lock.share();

  joint_entry_type expected = { EMPTY, EMPTY };
  joint_entry_type desired = { key, value };

  size_t i = hash(key);
  while (!entries[i].joint.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected.key != key) {
    i = (i + 1) & (nentries - 1);
    expected = {EMPTY, EMPTY};
  }

  value_type result;
  if (expected.key == key) {
    unreserve();  // key exists, cancel reservation for insert
    result = expected.value;
    key->decMemo();
    value->decShared();
  } else {
    result = value;
  }
  lock.unshare();
  return result;
}

bi::Map::value_type bi::Map::uninitialized_put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  reserve();
  lock.share();

  joint_entry_type expected = { EMPTY, EMPTY };
  joint_entry_type desired = { key, value };

  size_t i = hash(key);
  while (!entries[i].joint.compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected.key != key) {
    i = (i + 1) & (nentries - 1);
    expected = {EMPTY, EMPTY};
  }

  value_type result;
  if (expected.key == key) {
    unreserve();  // key exists, cancel reservation for insert
    result = expected.value;
  } else {
    result = value;
  }
  lock.unshare();
  return result;
}

void bi::Map::remove(const key_type key) {
  /* pre-condition */
  assert(key);

  if (!empty()) {
    lock.share();

    key_type expected = key;
    key_type desired = ERASED;

    size_t i = hash(key);
    while (!entries[i].split.key.compare_exchange_strong(expected, desired,
        std::memory_order_relaxed) && expected != EMPTY) {
      i = (i + 1) & (nentries - 1);
      expected = key;
    }
    if (expected == key) {
      value_type value = entries[i].split.value.load(
          std::memory_order_relaxed);
      lock.unshare();  // release first, as dec may cause lengthy cleanup
      key->decMemo();
      value->decShared();
    } else {
      lock.unshare();
    }
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
  size_t noccupied1 = noccupied.fetch_add(1u, std::memory_order_relaxed) + 1u;
  if (noccupied1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (noccupied1 > crowd()) {
      /* save previous table */
      size_t nentries1 = nentries;
      joint_entry_type* entries1 = (joint_entry_type*)entries;

      /* initialize new table */
      size_t nentries2 = std::max(2ull * nentries1, INITIAL_MAP_SIZE);
      joint_entry_type* entries2 = (joint_entry_type*)allocate(
          nentries2 * sizeof(entry_type));
      std::memset(entries2, 0, nentries2 * sizeof(entry_type));

      /* copy contents from previous table */
      size_t nerased = 0;
      nentries = nentries2;  // set this here as needed by hash()
      for (size_t i = 0u; i < nentries1; ++i) {
        joint_entry_type entry = entries1[i];
        if (entry.key == ERASED) {
          ++nerased;
        } else if (entry.key != EMPTY) {
          /* rehash and insert */
          size_t j = hash(entry.key);
          while (entries2[j].key) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          entries2[j] = entry;
        }
      }

      /* update object */
      entries = (entry_type*)entries2;
      noccupied -= nerased;

      /* deallocate previous table */
      deallocate(entries1, nentries1 * sizeof(joint_entry_type));
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Map::unreserve() {
  noccupied.fetch_sub(1u, std::memory_order_relaxed);
}

void bi::Map::clean() {
  lock.share();
  key_type key;
  value_type value;
  for (size_t i = 0u; i < nentries; ++i) {
    key = entries[i].split.key.load(std::memory_order_relaxed);
    if (key != EMPTY && key != ERASED && !key->isReachable()) {
      /* key is only reachable through this entry, so remove it */
      key_type expected = key;
      key_type desired = ERASED;
      if (entries[i].split.key.compare_exchange_strong(expected, desired,
          std::memory_order_relaxed)) {
        value = entries[i].split.value;
        key->decMemo();
        value->decShared();
      }
    }
  }
  lock.unshare();
}

void bi::Map::freeze() {
  lock.share();
  key_type key;
  value_type value;
  for (size_t i = 0u; i < nentries; ++i) {
    key = entries[i].split.key.load(std::memory_order_relaxed);
    if (key != EMPTY && key != ERASED) {
      value = entries[i].split.value.load(std::memory_order_relaxed);
      if (key->isReachable()) {
        value->freeze();
      } else {
        /* clean as we go */
        key_type expected = key;
        key_type desired = ERASED;
        if (entries[i].split.key.compare_exchange_strong(expected, desired,
            std::memory_order_relaxed)) {
          key->decMemo();
          value->decShared();
        }
      }
    }
  }
  lock.unshare();
}
