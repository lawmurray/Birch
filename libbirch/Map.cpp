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
    auto i = hash(key);
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

  auto i = hash(key);
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

  auto i = hash(key);
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

void libbirch::Map::freeze() {
  for (auto i = 0u; i < nentries; ++i) {
    auto v = values[i];
    if (v) {
      v->freeze();
    }
  }
}

void libbirch::Map::copy(Map& o) {
  assert(empty());

  /* resize */
  auto nentries1 = o.nentries;
  while (nentries1 / 2u > o.crowd()) {
    nentries1 /= 2u;
  }
  resize(nentries1);

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
}

void libbirch::Map::reserve() {
  ++noccupied;
  if (noccupied > crowd()) {
    /* save previous table */
    auto nentries1 = nentries;
    auto keys1 = keys;
    auto values1 = values;

    /* initialize new table */
    auto nentries2 = std::max(2u * nentries1,
        (unsigned)CLONE_MEMO_INITIAL_SIZE);
    auto keys2 = (key_type*)allocate(nentries2 * sizeof(key_type));
    auto values2 = (value_type*)allocate(nentries2 * sizeof(value_type));
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
          --noccupied;
        }
      }
    }

    /* update object */
    keys = keys2;
    values = values2;

    /* deallocate previous table */
    if (nentries1 > 0) {
      deallocate(keys1, nentries1 * sizeof(key_type), tentries);
      deallocate(values1, nentries1 * sizeof(value_type), tentries);
    }
    tentries = libbirch::tid;
  }
}

void libbirch::Map::resize(const unsigned nentries) {
  assert(empty());

  if (nentries > 0) {
    keys = (key_type*)allocate(nentries * sizeof(key_type));
    values = (value_type*)allocate(nentries * sizeof(value_type));
    std::memset(keys, 0, nentries * sizeof(key_type));
    std::memset(values, 0, nentries * sizeof(value_type));
    this->nentries = nentries;
    tentries = libbirch::tid;
  }
}
