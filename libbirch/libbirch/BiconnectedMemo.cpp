/**
*@file
 */
#include "libbirch/BiconnectedMemo.hpp"

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Any.hpp"

libbirch::BiconnectedMemo::BiconnectedMemo(Any* o) :
    values(nullptr),
    offset(o->k_),
    nentries(o->n_) {
  if (nentries > 0) {
    values = (Any**)std::malloc(nentries*sizeof(Any*));
    std::memset(values, 0, nentries*sizeof(Any*));
  }
}

libbirch::BiconnectedMemo::~BiconnectedMemo() {
  /* the entire array should have been used */
  assert(std::all_of(values, values + nentries, [](Any* o) {
        return o != nullptr;
      }));
  if (nentries > 0) {
    std::free(values);
  }
}

libbirch::Any*& libbirch::BiconnectedMemo::get(Any* key) {
  assert(key);
  int k = key->k_ + key->n_ - offset - 1;  // rank in biconnected component
  assert(0 <= k && k < nentries);
  return values[k];
}
