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
  while (k) {
    assert(k != key);
    i = (i + 1u) & (nentries - 1u);
    k = keys[i];
  }
  keys[i] = key;
  values[i] = value;
  return value;
}

libbirch::Map::value_type libbirch::Map::uninitialized_put(const key_type key,
    const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  reserve();
  auto i = hash(key, nentries);
  auto k = keys[i];
  while (k) {
    assert(k != key);
    i = (i + 1u) & (nentries - 1u);
    k = keys[i];
  }
  keys[i] = key;
  values[i] = value;
  return value;
}

void libbirch::Map::copy(Map& o) {
  assert(empty());

  /* allocate */
  nentries = o.nentries;
  if (nentries > 0) {
    keys = (key_type*)allocate(nentries * sizeof(key_type));
    values = (value_type*)allocate(nentries * sizeof(value_type));
    tentries = libbirch::tid;
  }
  noccupied = o.noccupied;

  /* copy */
  for (auto i = 0u; i < nentries; ++i) {
    auto key = o.keys[i];
    auto value = o.values[i];
    if (key) {
      /* apply the map to itself once to remove any long chains */
      value = o.get(value, value);

      key->incMemo();
      value->incShared();
    }
    keys[i] = key;
    values[i] = value;
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
    rehash();
  }
}

void libbirch::Map::rehash() {
  /* save previous table */
  auto nentries1 = nentries;
  auto tentries1 = tentries;
  auto keys1 = keys;
  auto values1 = values;

  /* first pass, count number of active entries */
  for (auto i = 0u; i < nentries; ++i) {
    auto key = keys[i];
    if (key && !key->isReachable()) {
      --noccupied;
    }
  }

  /* choose an appropriate size */
  nentries = std::max(2u*nentries, (unsigned)CLONE_MEMO_INITIAL_SIZE);
  while (noccupied < (nentries >> 2u) && nentries > (unsigned)CLONE_MEMO_INITIAL_SIZE) {
    nentries >>= 1u;
  }

  /* initialize new table */
  keys = (key_type*)allocate(nentries * sizeof(key_type));
  values = (value_type*)allocate(nentries * sizeof(value_type));
  std::memset(keys, 0, nentries * sizeof(key_type));
  std::memset(values, 0, nentries * sizeof(value_type));
  tentries = libbirch::tid;

  /* copy active entries from previous table */
  for (auto i = 0u; i < nentries1; ++i) {
    auto key = keys1[i];
    if (key && key->isReachable()) {
      auto value = values1[i];
      keys1[i] = nullptr;
      values1[i] = nullptr;  // not necessary
      auto j = hash(key, nentries);
      while (keys[j]) {
        j = (j + 1u) & (nentries - 1u);
      }
      keys[j] = key;
      values[j] = value;
    }
  }

  /* clean up inactive entries in previous table */
  for (auto i = 0u; i < nentries1; ++i) {
    auto key = keys1[i];
    if (key) {
      auto value = values1[i];
      key->decMemo();
      value->decShared();
    }
  }

  /* deallocate previous table */
  if (nentries1 > 0) {
    deallocate(keys1, nentries1 * sizeof(key_type), tentries1);
    deallocate(values1, nentries1 * sizeof(value_type), tentries1);
  }
}
