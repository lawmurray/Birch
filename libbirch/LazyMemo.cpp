/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "LazyMemo.hpp"

libbirch::LazyMemo::LazyMemo() :
    entries(nullptr),
    nentries(0u),
    tentries(0u),
    noccupied(0u),
    nnew(0u) {
  //
}

libbirch::LazyMemo::~LazyMemo() {
  if (nentries > 0u) {
    for (unsigned i = 0u; i < nentries; ++i) {
      auto key = entries[i].key;
      if (key) {
        auto value = entries[i].value;
        key->decMemo();
        value->doubleDecShared();
      }
    }
    deallocate(entries, nentries * sizeof(entry_type), tentries);
  }
}

libbirch::LazyMemo::value_type libbirch::LazyMemo::get(const key_type key,
    const value_type failed) {
  /* pre-condition */
  assert(key);

  auto value = failed;
  if (!empty()) {
    auto i = hash(key, nentries);
    auto k = entries[i].key;
    while (k && k != key) {
      i = (i + 1u) & (nentries - 1u);
      k = entries[i].key;
    }
    if (k == key) {
      value = entries[i].value;
    }
  }
  return value;
}
void libbirch::LazyMemo::put(const key_type key, const value_type value) {
  /* pre-condition */
  assert(key);
  assert(value);

  key->incMemo();
  value->doubleIncShared();

  reserve();
  auto i = hash(key, nentries);
  auto k = entries[i].key;
  while (k) {
    assert(k != key);
    i = (i + 1u) & (nentries - 1u);
    k = entries[i].key;
  }
  entries[i] = {key, value};
}

void libbirch::LazyMemo::copy(LazyMemo& o) {
  assert(empty());

  /* strategy here is to rehash the parent, which may reduce its size and
   * remove unreachable entries, then just copy entry-by-entry into
   * this, with no need to rehash */
  o.rehash();
  if (o.nentries > 0u) {
    /* allocate */
    entries = (entry_type*)allocate(o.nentries * sizeof(entry_type));
    nentries = o.nentries;
    tentries = libbirch::tid;
    noccupied = o.noccupied;
    nnew = o.nnew;

    /* copy entry-by-entry, incrementing reference counts for non-null
     * entries */
    for (auto i = 0u; i < nentries; ++i) {
      auto entry = o.entries[i];
      auto key = entry.key;
      auto value = entry.value;
      if (key) {
        key->incMemo();
      }
      if (value) {
        value->doubleIncShared();
      }
      entries[i] = {key, value};
    }
  }
}

void libbirch::LazyMemo::freeze() {
  for (auto i = 0u; i < nentries; ++i) {
    auto value = entries[i].value;
    if (value) {
      value->freeze();
    }
  }
}

void libbirch::LazyMemo::reserve() {
  ++nnew;
  ++noccupied;
  if (noccupied > crowd()) {
    rehash();
  }
}

void libbirch::LazyMemo::rehash() {
  if (nnew > 0u) {  // no need to rehash if no new entries since last time
    nnew = 0u;

    /* first pass, apply the table to itself; this has the effect of
     * replacing a -> b and b -> c with a -> c and b -> c, which may allow
     * b to be collected sooner */
    for (auto i = 0u; i < nentries; ++i) {
      auto key = entries[i].key;
      if (key) {
        auto first = entries[i].value;
        auto prev = first;
        auto next = first;
        do {
          prev = next;
          next = get(prev, prev);
        } while (next != prev);
        if (next != first) {
          next->doubleIncShared();
          first->doubleDecShared();
        }
        entries[i].value = next;
      }
    }

    /* second pass, delete any entries where the key is no longer reachable;
     * from this point, the old buffers are no long valid as a hash table */
    for (auto i = 0u; i < nentries; ++i) {
      auto key = entries[i].key;
      if (key && !key->isReachable()) {
        auto value = entries[i].value;
        key->decMemo();
        value->doubleDecShared();
        entries[i] = {nullptr, nullptr};
        --noccupied;
      }
    }

    /* save previous table */
    auto nentries1 = nentries;
    auto tentries1 = tentries;
    auto entries1 = entries;

    if (noccupied == 0u) {
      /* new table will be empty */
      nentries = 0u;
      tentries = 0u;
      entries = nullptr;
    } else {
      /* choose an appropriate size for the new table */
      unsigned minSize = (unsigned)CLONE_MEMO_INITIAL_SIZE;
      nentries = std::max(2u*nentries1, minSize);
      while (minSize < nentries && noccupied <= crowd()/2) {
        nentries /= 2u;
      }

      /* allocate the new table */
      entries = (entry_type*)allocate(nentries * sizeof(entry_type));
      std::memset(entries, 0, nentries * sizeof(entry_type));
      tentries = libbirch::tid;

      /* copy entries from previous table */
      for (auto i = 0u; i < nentries1; ++i) {
        auto key = entries1[i].key;
        if (key) {
          auto value = entries1[i].value;
          auto j = hash(key, nentries);
          while (entries[j].key) {
            j = (j + 1u) & (nentries - 1u);
          }
          entries[j] = {key, value};
        }
      }
    }

    /* deallocate previous table */
    if (nentries1 > 0) {
      deallocate(entries1, nentries1 * sizeof(entry_type), tentries1);
    }
  }
}

#endif
