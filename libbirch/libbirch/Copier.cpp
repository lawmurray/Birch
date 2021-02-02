/**
 * @file
 */
#include "libbirch/Copier.hpp"

libbirch::Any* libbirch::Copier::visit(Any* o) {
  int n = o->n;
  if (n > 0 && m.empty()) {
    m.resize(n, nullptr);
  }
  if (!m[n - 1]) {
    m[n - 1] = o->copy();
    m[n - 1]->accept_(*this);
  }
  return m[n - 1];
}
