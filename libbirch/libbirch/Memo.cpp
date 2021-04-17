/**
*@file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"

libbirch::Memo::Memo() :
    keys(nullptr),
    values(nullptr),
    nentries(0),
    noccupied(0) {
  //
}

libbirch::Memo::~Memo() {
  std::free(keys);
  std::free(values);
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
    values[i] = nullptr;
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
  keys = (Any**)std::malloc(nentries*sizeof(Any*));
  values = (Any**)std::malloc(nentries*sizeof(Any*));
  std::memset(keys, 0, nentries*sizeof(Any*));
  //std::memset(values, 0, nentries*sizeof(Any*));
  // ^ nullptr keys are used to indicate empty slots, while individual values
  //   are set to nullptr on first access in get(), so need not be here */

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
    std::free(keys1);
    std::free(values1);
  }
}
