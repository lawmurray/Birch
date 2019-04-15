/**
 * @file
 */
#include "libbirch/Map.hpp"

/**
 * Within Map, this is the value assigned to the key of an empty entry; one
 * which has never been used.
 */
static libbirch::Any* const EMPTY = nullptr;

/**
 * Within Map, this is the value assigned to the key of an erased entry; one
 * which was at some point used, but has since been erased. Such entries are
 * skipped when reading, but may be overwritten when writing.
 */
static libbirch::Any* const ERASED =
    reinterpret_cast<libbirch::Any*>(0xFFFFFFFFFFFFFFFF);

libbirch::Map::Map() :
    keys(nullptr),
    values(nullptr),
    nentries(0u),
    tentries(0u),
    noccupied(0u) {
  //
}

libbirch::Map::~Map() {
  if (nentries > 0u) {
    /* don't need thread safety here, so cast away the atomicity */
    key_type* keys1 = (key_type*)keys;
    value_type* values1 = (value_type*)values;
    key_type key;
    value_type value;
    for (unsigned i = 0u; i < nentries; ++i) {
      key = keys1[i];
      if (key != EMPTY && key != ERASED) {
        value = values1[i];
        key->decMemo();
        value->decShared();
      }
    }
    deallocate(keys1, nentries * sizeof(key_type), tentries);
    deallocate(values1, nentries * sizeof(value_type), tentries);
  }
}

libbirch::Map::value_type libbirch::Map::get(const key_type key,
    const value_type failed) {
  /* pre-condition */
  assert(key);

  value_type value = failed;
  if (!empty()) {
    lock.share();
    auto code = hash(key);
    auto i = code;
    auto k = keys[i].load(std::memory_order_relaxed);
    if (k && k != key) {
      i = (i + 1u) & (nentries - 1u);
      k = keys[i].load(std::memory_order_relaxed);
    }
    while (k && k != key && i != code) {
      // ^ extra condition i != code ensures we loop over once at most,
      //   happens when key not found, and all empty slots have keys of
      //   ERASED not EMPTY
      i = (i + 1u) & (nentries - 1u);
      k = keys[i].load(std::memory_order_relaxed);
    }
    if (k == key) {
      value = get(i);
    }
    lock.unshare();
  }
  return value;
}

libbirch::Map::value_type libbirch::Map::put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  key->incMemo();
  value->incShared();

  reserve();
  lock.share();

  key_type expected = EMPTY;
  key_type desired = key;

  auto i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected != key) {
    if (expected == ERASED) {
      // attempt to replace this entry on next attempt
    } else {
      i = (i + 1u) & (nentries - 1u);
      expected = EMPTY;
    }
  }

  value_type result;
  if (expected == key) {
    /* key exists, cancel put and return associated value */
    unreserve();
    key->decMemo();
    value->decShared();
    result = get(i);
  } else {
    values[i].store(value, std::memory_order_relaxed);
    result = value;
  }
  lock.unshare();
  return result;
}

libbirch::Map::value_type libbirch::Map::uninitialized_put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  reserve();
  lock.share();

  key_type expected = EMPTY;
  key_type desired = key;

  auto i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected != key) {
    if (expected == ERASED) {
      // attempt to replace this entry on next attempt
    } else {
      i = (i + 1u) & (nentries - 1u);
      expected = EMPTY;
    }
  }

  value_type result;
  if (expected == key) {
    unreserve();  // key exists, cancel reservation for insert
    result = get(i);
  } else {
    values[i].store(value, std::memory_order_relaxed);
    result = value;
  }
  lock.unshare();
  return result;
}

void libbirch::Map::freeze() {
  lock.share();
  for (auto i = 0u; i < nentries; ++i) {
    auto key = keys[i].load(std::memory_order_relaxed);
    if (key != EMPTY && key != ERASED) {
      if (key->isReachable()) {
        get(i)->freeze();
      } else {
        erase(i);
      }
    }
  }
  lock.unshare();
}

void libbirch::Map::copy(Map& o) {
  o.lock.keep();
  lock.keep();
  if (o.nentries > 0) {
    auto nentries1 = o.nentries;
    auto keys1 = (key_type*)allocate(nentries1 * sizeof(key_type));
    auto values1 = (value_type*)allocate(nentries1 * sizeof(value_type));
    for (auto i = 0u; i < nentries1; ++i) {
      auto key = o.keys[i].load(std::memory_order_relaxed);
      auto value = o.values[i].load(std::memory_order_relaxed);
      if (key != EMPTY && key != ERASED) {
        if (key->isReachable()) {
          key->incMemo();
          value->incShared();
        } else {
          o.erase(i);
          key = ERASED;
          value = EMPTY;
        }
      }
      keys1[i] = key;
      values1[i] = value;
    }

    keys = (std::atomic<key_type>*)keys1;
    values = (std::atomic<value_type>*)values1;
    nentries = nentries1;
    tentries = libbirch::tid;
    noccupied.store(o.noccupied.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
  o.lock.unkeep();
  lock.unkeep();
}

libbirch::Map::value_type libbirch::Map::get(const unsigned i) {
  /* key is written before value on put, so loop for a valid value in
   * case that write has not concluded yet */
  value_type value;
  do {
    value = values[i].load(std::memory_order_relaxed);
  } while (value == EMPTY);
  return value;
}

void libbirch::Map::erase(const unsigned i) {
  auto value = values[i].exchange(EMPTY, std::memory_order_relaxed);
  auto key = keys[i].exchange(ERASED, std::memory_order_relaxed);
  if (key != ERASED) {
    unreserve();
    key->decMemo();
  }
  if (value != EMPTY) {
    value->decShared();
  }
}

void libbirch::Map::reserve() {
  unsigned noccupied1 = noccupied.fetch_add(1u, std::memory_order_relaxed)
      + 1u;
  if (noccupied1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    noccupied1 = noccupied.load(std::memory_order_relaxed);
    if (noccupied1 > crowd()) {
      /* save previous table */
      auto nentries1 = nentries;
      key_type* keys1 = (key_type*)keys;
      value_type* values1 = (value_type*)values;

      /* initialize new table */
      unsigned nentries2 = std::max(2u * nentries1,
          (unsigned)CLONE_MEMO_INITIAL_SIZE);
      key_type* keys2 = (key_type*)allocate(nentries2 * sizeof(key_type));
      value_type* values2 = (value_type*)allocate(
          nentries2 * sizeof(value_type));
      std::memset(keys2, 0, nentries2 * sizeof(key_type));
      std::memset(values2, 0, nentries2 * sizeof(value_type));

      /* copy contents from previous table */
      nentries = nentries2;  // set this here as needed by hash()
      for (auto i = 0u; i < nentries1; ++i) {
        auto key = keys1[i];
        auto value = values1[i];
        if (key != EMPTY && key != ERASED) {
          if (key->isReachable()) {
            /* rehash and insert */
            auto j = hash(key);
            while (keys2[j]) {
              j = (j + 1u) & (nentries2 - 1u);
            }
            keys2[j] = key;
            values2[j] = value;
          } else {
            key->decMemo();
            value->decShared();
            --noccupied1;
          }
        }
      }

      /* update object */
      keys = (std::atomic<key_type>*)keys2;
      values = (std::atomic<value_type>*)values2;

      /* deallocate previous table */
      if (nentries1 > 0) {
        deallocate(keys1, nentries1 * sizeof(key_type), tentries);
        deallocate(values1, nentries1 * sizeof(value_type), tentries);
      }
      tentries = libbirch::tid;
      noccupied.store(noccupied1, std::memory_order_relaxed);
    }

    /* release resize lock */
    lock.unkeep();
  }
}
