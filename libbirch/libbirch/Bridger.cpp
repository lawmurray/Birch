/**
 * @file
 */
#include "libbirch/Bridger.hpp"

std::tuple<int,int,int,int> libbirch::Bridger::visit(const int j, const int k,
    Any* o) {
  if (o->p == get_thread_num()) {
    int l, h, m, n, l1, h1, m1, n1;
    o->p = -1;
    if (o->a < o->numShared()) {
      l = 0;
      h = MAX;
    } else {
      l = o->l;
      h = o->h;
    }
    std::tie(l1, h1, m1, n1) = o->accept_(*this, j + 1, k);
    l = std::min(l, l1);
    h = std::max(h, h1);
    m = m1 + 1;
    n = n1 + 1;

    o->n = k + n;  // post-order rank in biconnected component
    o->l = 0;
    o->h = 0;
    o->a = 0;
    o->f.maskAnd(static_cast<decltype(CLAIMED)>(~CLAIMED));
    // ^ ~ unary operator seems to convert uint8_t to int
    return std::make_tuple(l, h, m, n);
  } else {
    return std::make_tuple(MAX, 0, 0, 0);
  }
}
