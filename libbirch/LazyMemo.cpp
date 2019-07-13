/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "LazyMemo.hpp"

libbirch::LazyMemo::LazyMemo() :
    keys(nullptr),
    values(nullptr),
    nentries(0u),
    tentries(0u),
    noccupied(0u) {
  //
}

libbirch::LazyMemo::~LazyMemo() {
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

libbirch::LazyMemo::value_type libbirch::LazyMemo::get(const key_type key,
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
void libbirch::LazyMemo::put(const key_type key,
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
}

void libbirch::LazyMemo::copy(LazyMemo& o) {
  assert(empty());

  /* count number of active entries in parent */
  auto nactive = 0u;
  for (auto i = 0u; i < o.nentries; ++i) {
    auto key = o.keys[i];
    if (key && key->isReachable()) {
      ++nactive;
    }
  }

  if (nactive > 0u) {
    /* choose an appropriate size */
    nentries = std::max(o.nentries, (unsigned)CLONE_MEMO_INITIAL_SIZE);
    while (nactive < (nentries >> 2u) && nentries > (unsigned)CLONE_MEMO_INITIAL_SIZE) {
      nentries >>= 1u;
    }

    /* allocate */
    keys = (key_type*)allocate(nentries * sizeof(key_type));
    values = (value_type*)allocate(nentries * sizeof(value_type));
    std::memset(keys, 0, nentries * sizeof(key_type));
    std::memset(values, 0, nentries * sizeof(value_type));
    tentries = libbirch::tid;

    /* copy */
    for (auto i = 0u; i < o.nentries; ++i) {
      auto key = o.keys[i];
      if (key && key->isReachable()) {
        auto value = o.values[i];
        value = o.get(value, value);  // apply map once to shorten chains
        put(key, value);
      }
    }
  }
}

void libbirch::LazyMemo::freeze() {
  for (auto i = 0u; i < nentries; ++i) {
    auto v = values[i];
    if (v) {
      v->freeze();
    }
  }
}

void libbirch::LazyMemo::reserve() {
  if (++noccupied > crowd()) {
    rehash();
  }
}

void libbirch::LazyMemo::rehash() {
  /* save previous table */
  auto nentries1 = nentries;
  auto tentries1 = tentries;
  auto keys1 = keys;
  auto values1 = values;

  /* initialize new table */
  nentries = std::max(2u*nentries1, (unsigned)CLONE_MEMO_INITIAL_SIZE);
  keys = (key_type*)allocate(nentries * sizeof(key_type));
  values = (value_type*)allocate(nentries * sizeof(value_type));
  std::memset(keys, 0, nentries * sizeof(key_type));
  std::memset(values, 0, nentries * sizeof(value_type));
  tentries = libbirch::tid;

  /* copy entries from previous table */
  for (auto i = 0u; i < nentries1; ++i) {
    auto key = keys1[i];
    if (key) {
      auto j = hash(key, nentries);
      while (keys[j]) {
        j = (j + 1u) & (nentries - 1u);
      }
      keys[j] = key;
      values[j] = values1[i];
    }
  }

  /* deallocate previous table */
  if (nentries1 > 0) {
    deallocate(keys1, nentries1 * sizeof(key_type), tentries1);
    deallocate(values1, nentries1 * sizeof(value_type), tentries1);
  }
}

#endif
