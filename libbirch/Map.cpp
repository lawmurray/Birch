/**
 * @file
 */
#include "libbirch/Map.hpp"

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
      if (key) {
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
  lock.share();
  if (!empty()) {
    auto i = hash(key);
    auto k = keys[i].load(std::memory_order_relaxed);
    while (k && k != key) {
      i = (i + 1u) & (nentries - 1u);
      k = keys[i].load(std::memory_order_relaxed);
    }
    if (k == key) {
      value = get(i);
    }
  }
  lock.unshare();
  return value;
}

libbirch::Map::value_type libbirch::Map::put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  key->incMemo();
  value->incShared();

  key_type expected = nullptr;
  key_type desired = key;

  reserve();
  lock.share();

  auto i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected != key) {
    i = (i + 1u) & (nentries - 1u);
    expected = nullptr;
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

  key_type expected = nullptr;
  key_type desired = key;

  reserve();
  lock.share();

  auto i = hash(key);
  while (!keys[i].compare_exchange_strong(expected, desired,
      std::memory_order_relaxed) && expected != key) {
    i = (i + 1u) & (nentries - 1u);
    expected = nullptr;
  }

  value_type result;
  if (expected == key) {
    /* key exists, cancel put and return associated value */
    unreserve();
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
    if (key && key->isReachable()) {
      get(i)->freeze();
    }
  }
  lock.unshare();
}

void libbirch::Map::copy(Map& o) {
  o.lock.share();

  /* allocate buffers */
  auto nentries1 = o.nentries;
  while ((nentries1 >> 1u) > o.crowd()) {
    nentries1 >>= 1u;
  }
  nentries1 = std::max(nentries1, (unsigned)CLONE_MEMO_INITIAL_SIZE);
  resize(nentries1);

  /* copy entries */
  for (auto i = 0u; i < o.nentries; ++i) {
    auto key = o.keys[i].load(std::memory_order_relaxed);
    if (key && key->isReachable()) {
      auto value = o.values[i].load(std::memory_order_relaxed);
      put(key, value);
    }
  }
  o.lock.unshare();
}

libbirch::Map::value_type libbirch::Map::get(const unsigned i) {
  /* key is written before value on put, so loop for a valid value in
   * case that write has not concluded yet */
  value_type value;
  do {
    value = values[i].load(std::memory_order_relaxed);
  } while (!value);
  return value;
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
        if (key) {
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

void libbirch::Map::resize(const unsigned nentries) {
  assert(empty());

  if (nentries > 0) {
    lock.keep();

    auto keys = (key_type*)allocate(nentries*sizeof(key_type));
    auto values = (value_type*)allocate(nentries*sizeof(value_type));
    std::memset(keys, 0, nentries * sizeof(key_type));
    std::memset(values, 0, nentries * sizeof(value_type));

    this->keys = (std::atomic<key_type>*)keys;
    this->values = (std::atomic<value_type>*)values;
    this->nentries = nentries;
    tentries = libbirch::tid;

    lock.unkeep();
  }
}
