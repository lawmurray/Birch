/**
 * @file
 */
#include "libbirch/Copier.hpp"

libbirch::Copier::Copier() : offset(0) {
  //
}

libbirch::Copier::~Copier() {
  /* all elements of the memo should have been used */
  assert(std::all_of(m.begin(), m.end(), [](Any* o) {
        return o != nullptr;
      }));
}

libbirch::Any* libbirch::Copier::visit(Any* o) {
  int k = o->k;
  int n = o->n;
  if (m.empty()) {
    /* initialize */
    this->offset = k;
    this->m.resize(n, nullptr);
  }
  k = k + n - offset - 1;  // rank in biconnected component

  assert(0 <= k && k < m.size());
  if (!m[k]) {
    assert(!biconnected_copy());
    biconnected_copy(true);
    assert(biconnected_copy());
    m[k] = o->copy();
    biconnected_copy(true);
    assert(!biconnected_copy());
    m[k]->accept_(*this);
  }
  return m[k];
}
