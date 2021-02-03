/**
 * @file
 */
#include "libbirch/Copier.hpp"

libbirch::Any* libbirch::Copier::visit(Any* o) {
  int n = o->n;
  if (m.empty()) {
    m.resize(n, nullptr);
  }
  assert(n <= m.size());
  if (!m[n - 1]) {
    m[n - 1] = o->copy();
    m[n - 1]->accept_(*this);
  }
  return m[n - 1];
}
