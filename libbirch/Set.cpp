/**
 * @file
 */
#include "libbirch/Set.hpp"

static libbirch::Memo* const EMPTY = nullptr;
static const unsigned SET_INITIAL_SIZE = 4;

libbirch::Set::Set() :
    values(nullptr),
    nentries(0u),
    tentries(0u),
    noccupied(0u) {
  //
}

libbirch::Set::~Set() {
  if (nentries > 0) {
    deallocate(values, nentries * sizeof(value_type), tentries);
  }
}

bool libbirch::Set::contains(const value_type value) {
  bool result = false;
  if (!empty()) {
    lock.share();
    unsigned i = hash(value);
    value_type v = values[i].load(std::memory_order_relaxed);
    while (v && v != value) {
      i = (i + 1u) & (nentries - 1u);
      v = values[i].load(std::memory_order_relaxed);
    }
    result = (v == value);
    lock.unshare();
  }
  return result;
}

void libbirch::Set::insert(const value_type value) {
  /* pre-condition */
  assert(value);

  reserve();
  lock.share();
  value_type expected = EMPTY;
  value_type desired = value;
  unsigned i = hash(value);
  while (!values[i].compare_exchange_strong(expected, desired)
      && expected != value) {
    i = (i + 1u) & (nentries - 1u);
    expected = EMPTY;
  }
  if (expected == value) {
    unreserve();  // value exists, cancel reservation for insert
  }
  lock.unshare();
}

void libbirch::Set::reserve() {
  unsigned noccupied1 = noccupied.fetch_add(1u, std::memory_order_relaxed)
      + 1u;
  if (noccupied1 > crowd()) {
    /* obtain resize lock */
    lock.keep();

    /* check that no other thread has resized in the meantime */
    if (noccupied1 > crowd()) {
      /* save previous contents */
      unsigned nentries1 = nentries;
      value_type* values1 = (value_type*)values;

      /* initialize new contents */
      unsigned nentries2 = std::max(2u * nentries1, SET_INITIAL_SIZE);
      value_type* values2 = (value_type*)allocate(
          nentries2 * sizeof(value_type));
      std::memset(values2, 0, nentries2 * sizeof(value_type));

      /* copy contents from previous set */
      nentries = nentries2;  // set this here as needed by hash()
      for (unsigned i = 0u; i < nentries1; ++i) {
        value_type value = values1[i];
        if (value != EMPTY) {
          /* rehash and insert */
          unsigned j = hash(value);
          while (values2[j]) {
            j = (j + 1u) & (nentries2 - 1u);
          }
          values2[j] = value;
        }
      }

      /* update object */
      values = (std::atomic<value_type>*)values2;

      /* deallocate previous table */
      if (nentries1 > 0) {
        deallocate(values1, nentries1 * sizeof(value_type), tentries);
      }
      tentries = libbirch::tid;
    }

    /* release resize lock */
    lock.unkeep();
  }
}
