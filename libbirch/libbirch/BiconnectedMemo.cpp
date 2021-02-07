/**
*@file
 */
#include "libbirch/BiconnectedMemo.hpp"

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Any.hpp"

libbirch::BiconnectedMemo::BiconnectedMemo(Any* o) :
    values(nullptr),
    offset(o->k),
    nentries(o->n) {
  if (nentries > 0) {
    values = (Any**)allocate(nentries*sizeof(Any*));
    std::memset(values, 0, nentries*sizeof(Any*));
  }
}

libbirch::BiconnectedMemo::~BiconnectedMemo() {
  /* the entire array should have been used */
  assert(std::all_of(values, values + nentries, [](Any* o) {
        return o != nullptr;
      }));
  if (nentries > 0) {
    deallocate(values, nentries*sizeof(Any*), get_thread_num());
  }
}

libbirch::Any*& libbirch::BiconnectedMemo::get(Any* key) {
  assert(key);
  int k = key->k + key->n - offset - 1;  // rank in biconnected component
  assert(0 <= k && k < nentries);
  return values[k];
}
