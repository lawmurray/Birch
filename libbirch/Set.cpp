/**
 * @file
 */
#include "libbirch/Set.hpp"

static bi::Memo* const EMPTY = nullptr;

bi::Set::Set() :
    entries(nullptr),
    nentries(0),
    noccupied(0) {
  //
}

bi::Set::~Set() {
  deallocate(entries, nentries * sizeof(entry_type));
}

bool bi::Set::empty() const {
  return nentries == 0;
}

bool bi::Set::contains(const value_type value) {
  bool result = false;
  if (!empty()) {
    lock.share();
    size_t i = hash(value);
    value_type v = entries[i].load(std::memory_order_relaxed);
    while (v && v != value) {
      i = (i + 1) & (nentries - 1);
      v = entries[i].load(std::memory_order_relaxed);
    }
    result = (v == value);
    lock.unshare();
  }
  return result;
}

void bi::Set::insert(const value_type value) {
  /* pre-condition */
  assert(value);

  reserve();
  lock.share();
  value_type expected = EMPTY;
  value_type desired = value;
  size_t i = hash(value);
  while (!entries[i].compare_exchange_strong(expected, desired)
      && expected != value) {
    i = (i + 1) & (nentries - 1);
    expected = EMPTY;
  }
  if (expected == value) {
    unreserve();  // value exists, cancel reservation for insert
  }
  lock.unshare();
}

size_t bi::Set::hash(const value_type value) const {
  assert(nentries > 0);
  return (reinterpret_cast<size_t>(value) >> 5ull) & (nentries - 1ull);
}

size_t bi::Set::crowd() const {
  /* the set is considered crowded if more than three-quarters of its
   * entries are occupied */
  return (nentries >> 1ull) + (nentries >> 2ull);
}

void bi::Set::reserve() {
  size_t noccupied1 = ++noccupied;
  if (noccupied1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (noccupied1 > crowd()) {
      /* save previous contents */
      size_t nentries1 = nentries;
      value_type* entries1 = (value_type*)entries;

      /* initialize new contents */
      size_t nentries2 = std::max(2ull * nentries1, INITIAL_SET_SIZE);
      value_type* entries2 = (value_type*)allocate(
          nentries2 * sizeof(entry_type));
      std::memset(entries2, 0, nentries2 * sizeof(entry_type));

      /* copy contents from previous set */
      nentries = nentries2;  // set this here as needed by hash()
      for (size_t i = 0u; i < nentries1; ++i) {
        value_type value = entries1[i];
        if (value != EMPTY) {
          /* rehash and insert */
          size_t j = hash(value);
          while (entries2[j]) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          entries2[j] = value;
        }
      }

      /* update object */
      entries = (entry_type*)entries2;

      /* deallocate previous table */
      deallocate(entries1, nentries1 * sizeof(entry_type));
    }

    /* release resize lock */
    lock.unkeep();
  }
}

void bi::Set::unreserve() {
  --noccupied;
}
