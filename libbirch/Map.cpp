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
    key_type key;
    value_type value;
    for (unsigned i = 0u; i < nentries; ++i) {
      key = keys[i];
      if (key) {
        value = values[i];
        key->decMemo();
        value->decShared();
      }
    }
    deallocate(keys, nentries * sizeof(key_type), tentries);
    deallocate(values, nentries * sizeof(value_type), tentries);
  }
}

libbirch::Map::value_type libbirch::Map::get(const key_type key,
    const value_type failed) {
  /* pre-condition */
  assert(key);

  value_type value = failed;
  if (!empty()) {
    auto i = hash(key, nentries);
    auto k = keys[i];
    while (k && k != key) {
      i = (i + 1u) & (nentries - 1u);
      k = keys[i];
    }
    if (k == key) {
      value = values[i];
    }
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

  auto i = hash(key, nentries);
  auto k = keys[i];
  while (k && k != key) {
    i = (i + 1u) & (nentries - 1u);
    k = keys[i];
  }

  value_type result;
  if (k == key) {
    /* key exists, cancel put and return associated value */
    unreserve();
    key->decMemo();
    value->decShared();
    result = values[i];
  } else {
    keys[i] = key;
    values[i] = value;
    result = value;
  }
  return result;
}

libbirch::Map::value_type libbirch::Map::uninitialized_put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  reserve();

  auto i = hash(key, nentries);
  auto k = keys[i];
  while (k && k != key) {
    i = (i + 1u) & (nentries - 1u);
    k = keys[i];
  }

  value_type result;
  if (k == key) {
    /* key exists, cancel put and return associated value */
    unreserve();
    result = values[i];
  } else {
    keys[i] = key;
    values[i] = value;
    result = value;
  }
  return result;
}

void libbirch::Map::copy(Map& o) {
  assert(empty());

  /* allocate */
  nentries = o.nentries;
  if (nentries > 0) {
    keys = (key_type*)allocate(nentries * sizeof(key_type));
    values = (value_type*)allocate(nentries * sizeof(value_type));
    std::memset(keys, 0, nentries * sizeof(key_type));
    std::memset(values, 0, nentries * sizeof(value_type));
    tentries = libbirch::tid;
  }

  /* copy */
  for (auto i = 0u; i < o.nentries; ++i) {
    auto key = o.keys[i];
    if (key && key->isReachable()) {
      auto value = o.values[i];
      value = o.get(value, value);
      // ^ applies the existing map to itself once, taking one step toward
      //   its transitive closure, and eliminating long chains of mappings
      //   that keep obsolete objects in scope
      put(key, value);
    }
  }

  /* shrink if possible */
  auto nentries2 = nentries;
  while (noccupied < (nentries2 >> 2u) && nentries2 > (unsigned)CLONE_MEMO_INITIAL_SIZE) {
    nentries2 >>= 1u;
  }
  if (nentries2 != nentries) {
    resize(nentries2);
  }
}

#if ENABLE_LAZY_DEEP_CLONE
void libbirch::Map::freeze() {
  for (auto i = 0u; i < nentries; ++i) {
    auto v = values[i];
    if (v) {
      v->freeze();
    }
  }
}
#endif

void libbirch::Map::reserve() {
  if (++noccupied > crowd()) {
    resize(std::max(2u * nentries, (unsigned)CLONE_MEMO_INITIAL_SIZE));
  }
}

void libbirch::Map::resize(const unsigned nentries2) {
  /* save previous table */
  auto nentries1 = nentries;
  auto keys1 = keys;
  auto values1 = values;

  /* initialize new table */
  auto keys2 = (key_type*)allocate(nentries2 * sizeof(key_type));
  auto values2 = (value_type*)allocate(nentries2 * sizeof(value_type));
  std::memset(keys2, 0, nentries2 * sizeof(key_type));
  std::memset(values2, 0, nentries2 * sizeof(value_type));

  /* copy contents from previous table */
  for (auto i = 0u; i < nentries1; ++i) {
    auto key = keys1[i];
    if (key) {
      /* adding a key->isReachable() check to remove obsolete entries while
       * doing this is causing sporadic thread safety problems that are not
       * fully understood */
      auto value = values1[i];
      auto j = hash(key, nentries2);
      while (keys2[j]) {
        j = (j + 1u) & (nentries2 - 1u);
      }
      keys2[j] = key;
      values2[j] = value;
    }
  }

  /* update object */
  nentries = nentries2;
  keys = keys2;
  values = values2;

  /* deallocate previous table */
  if (nentries1 > 0) {
    deallocate(keys1, nentries1 * sizeof(key_type), tentries);
    deallocate(values1, nentries1 * sizeof(value_type), tentries);
  }
  tentries = libbirch::tid;
}
