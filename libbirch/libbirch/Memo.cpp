/**
*@file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"

libbirch::Memo::Memo() :
    keys(nullptr),
    values(nullptr),
    nentries(0),
    noccupied(0) {
  //
}

libbirch::Memo::~Memo() {
  auto tid = get_thread_num();
  deallocate(keys, nentries*sizeof(Any*), tid);
  deallocate(values, nentries*sizeof(Any*), tid);
}

libbirch::Any*& libbirch::Memo::get(Any* key) {
  assert(key);

  /* reserve a slot */
  if (++noccupied > crowd()) {
    rehash();
  }

  auto i = hash(key);
  auto k = keys[i];
  while (k && k != key) {
    i = (i + 1) & (nentries - 1);
    k = keys[i];
  }
  if (k) {
    --noccupied;  // unreserve the slot, wasn't needed
  } else {
    keys[i] = key;
  }
  return values[i];
}

void libbirch::Memo::rehash() {
  /* save previous table */
  auto nentries1 = nentries;
  auto keys1 = keys;
  auto values1 = values;

  /* size of new table */
  nentries = std::max(INITIAL_SIZE, 2*nentries1);

  /* allocate the new table */
  keys = (Any**)allocate(nentries*sizeof(Any*));
  values = (Any**)allocate(nentries*sizeof(Any*));
  std::memset(keys, 0, nentries*sizeof(Any*));
  std::memset(values, 0, nentries*sizeof(Any*));

  /* copy entries from previous table */
  for (int i = 0; i < nentries1; ++i) {
    auto key = keys1[i];
    if (key) {
      auto j = hash(key);
      while (keys[j]) {
        j = (j + 1) & (nentries - 1);
      }
      keys[j] = key;
      values[j] = values1[i];
    }
  }

  /* deallocate previous table */
  if (nentries1 > 0) {
    auto tid = get_thread_num();
    deallocate(keys1, nentries1*sizeof(Any*), tid);
    deallocate(values1, nentries1*sizeof(Any*), tid);
  }
}
