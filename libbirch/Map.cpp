/**
 * @file
 */
#include "libbirch/Map.hpp"

static bi::Any* const EMPTY = nullptr;
static bi::Any* const ERASED = reinterpret_cast<bi::Any*>(0xFFFFFFFFFFFFFFFF);

bi::Map::Map() :
    keys(nullptr),
    values(nullptr),
    nentries(0u),
    tentries(0u),
    noccupied(0u) {
  //
}

bi::Map::~Map() {
  if (nentries > 0) {
    key_type key;
    value_type value;
    for (unsigned i = 0u; i < nentries; ++i) {
      ///@todo As with resize, could cast to non-atomic here
      key = keys[i].load(std::memory_order_seq_cst);
      if (key != EMPTY && key != ERASED) {
        value = values[i].load(std::memory_order_seq_cst);
        key->decMemo();
        value->decShared();
      }
    }
    deallocate(keys, nentries * sizeof(key_type), tentries);
    deallocate(values, nentries * sizeof(value_type), tentries);
  }
}

bi::Map::value_type bi::Map::get(const key_type key,
    const value_type failed) {
  /* pre-condition */
  assert(key);

  value_type value = failed;
  if (!empty()) {
    lock.share();
    unsigned i = hash(key);
    key_type k = keys[i].load(std::memory_order_seq_cst);
    while (k && k != key) {
      i = (i + 1u) & (nentries - 1u);
      k = keys[i].load(std::memory_order_seq_cst);
    }
    if (k == key) {
      value = get(i);
    }
    lock.unshare();
  }
  return value;
}

bi::Map::value_type bi::Map::get(const unsigned i) {
  /* key is written before value on put, so loop for a valid value in
   * case that write has not concluded yet */
  value_type value;
  do {
    value = values[i].load(std::memory_order_seq_cst);
  } while (value == EMPTY);
  return value;
}

bi::Map::value_type bi::Map::put(const key_type key, const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  key->incMemo();
  value->incShared();

  reserve();
  lock.share();

  key_type expected = EMPTY;
  key_type desired = key;

  unsigned i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_seq_cst) && expected != key) {
    i = (i + 1u) & (nentries - 1u);
    expected = EMPTY;
  }

  value_type result;
  if (expected == key) {
    /* key exists, cancel put and return associated value */
    unreserve();
    key->decMemo();
    value->decShared();
    result = get(i);
  } else {
    values[i].store(value, std::memory_order_seq_cst);
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

  key_type expected = EMPTY;
  key_type desired = key;

  unsigned i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_seq_cst) && expected != key) {
    i = (i + 1u) & (nentries - 1u);
    expected = EMPTY;
  }

  value_type result;
  if (expected == key) {
    unreserve();  // key exists, cancel reservation for insert
    result = get(i);
  } else {
    values[i].store(value, std::memory_order_seq_cst);
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

    unsigned i = hash(key);
    while (!keys[i].compare_exchange_strong(expected, desired,
        std::memory_order_seq_cst) && expected != EMPTY) {
      i = (i + 1u) & (nentries - 1u);
      expected = key;
    }
    if (expected == key) {
      value_type value = values[i].load(std::memory_order_seq_cst);
      lock.unshare();  // release first, as dec may cause lengthy cleanup
      key->decMemo();
      value->decShared();
    } else {
      lock.unshare();
    }
  }
}

void bi::Map::reserve() {
  unsigned noccupied1 = noccupied.fetch_add(1u, std::memory_order_seq_cst)
      + 1u;
  if (noccupied1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (noccupied1 > crowd()) {
      /* save previous table */
      unsigned nentries1 = nentries;
      key_type* keys1 = (key_type*)keys;
      value_type* values1 = (value_type*)values;

      /* initialize new table */
      unsigned nentries2 = std::max(2u * nentries1, INITIAL_MAP_SIZE);
      key_type* keys2 = (key_type*)allocate(nentries2 * sizeof(key_type));
      value_type* values2 = (value_type*)allocate(
          nentries2 * sizeof(value_type));
      std::memset(keys2, 0, nentries2 * sizeof(key_type));
      std::memset(values2, 0, nentries2 * sizeof(value_type));

      /* copy contents from previous table */
      unsigned nerased = 0u;
      nentries = nentries2;  // set this here as needed by hash()
      for (unsigned i = 0u; i < nentries1; ++i) {
        key_type key = keys1[i];
        if (key == ERASED) {
          ++nerased;
        } else if (key != EMPTY) {
          /* rehash and insert */
          unsigned j = hash(key);
          while (keys2[j]) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          keys2[j] = key;
          values2[j] = values1[i];
        }
      }

      /* update object */
      keys = (std::atomic<key_type>*)keys2;
      values = (std::atomic<value_type>*)values2;
      noccupied -= nerased;

      /* deallocate previous table */
      if (nentries1 > 0) {
        deallocate(keys1, nentries1 * sizeof(key_type), tentries);
        deallocate(values1, nentries1 * sizeof(value_type), tentries);
      }
      tentries = bi::tid;
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Map::clean() {
  lock.share();
  key_type key;
  value_type value;
  for (unsigned i = 0u; i < nentries; ++i) {
    key = keys[i].load(std::memory_order_seq_cst);
    if (key != EMPTY && key != ERASED && !key->isReachable()) {
      /* key is only reachable through this entry, so remove it */
      key_type expected = key;
      key_type desired = ERASED;
      if (keys[i].compare_exchange_strong(expected, desired,
          std::memory_order_seq_cst)) {
        value = values[i].load(std::memory_order_seq_cst);
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
  for (unsigned i = 0u; i < nentries; ++i) {
    key = keys[i].load(std::memory_order_seq_cst);
    if (key != EMPTY && key != ERASED) {
      value = values[i].load(std::memory_order_seq_cst);
      if (key->isReachable()) {
        value->freeze();
      } else {
        /* clean as we go */
        key_type expected = key;
        key_type desired = ERASED;
        if (keys[i].compare_exchange_strong(expected, desired,
            std::memory_order_seq_cst)) {
          key->decMemo();
          value->decShared();
        }
      }
    }
  }
  lock.unshare();
}
