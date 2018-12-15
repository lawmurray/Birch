/**
 * @file
 */
#include "libbirch/List.hpp"

bi::List::List() :
    entries(nullptr),
    nentries(0),
    noccupied(0) {
  //
}

bi::List::~List() {
  for (size_t i = 0; i < noccupied; ++i) {
    entries[i]->releaseMemo();
    entries[i]->decWeak();
  }
  deallocate(entries, nentries * sizeof(value_type));
}

bool bi::List::empty() const {
  return nentries == 0;
}

void bi::List::put(const value_type value) {
  value->incWeak();
  size_t i = reserve();
  lock.share();
  entries[i] = value;
  lock.unshare();
}

size_t bi::List::reserve() {
  size_t noccupied1 = noccupied++;
  if (noccupied1 >= nentries) {
    lock.keep();
    size_t nentries1 = std::max(2ull * nentries, INITIAL_LIST_SIZE);
    if (entries) {
      entries = (value_type*)reallocate(entries,
          nentries * sizeof(value_type), nentries1 * sizeof(value_type));
    } else {
      entries = (value_type*)allocate(nentries1 * sizeof(value_type));
    }
    nentries = nentries1;
    lock.unkeep();
  }
  return noccupied1;
}
