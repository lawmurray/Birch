/**
 * @file
 */
#include "libbirch/Bridger.hpp"

std::tuple<int,int,int,int> libbirch::Bridger::visit(const int j, const int k,
    Any* o) {
  if (o->p_ == get_thread_num()) {
    int l, h, m, n, l1, h1, m1, n1;
    o->p_ = -1;
    if (o->a_ < o->numShared_()) {
      l = 0;
      h = MAX;
    } else {
      l = o->l_;
      h = o->h_;
    }
    std::tie(l1, h1, m1, n1) = o->accept_(*this, j + 1, k);
    l = std::min(l, l1);
    h = std::max(h, h1);
    m = m1 + 1;
    n = n1 + 1;
    
    o->a_ = 0;
    o->k_ = k;
    o->n_ = n;
    o->f_.maskAnd(~(CLAIMED|POSSIBLE_ROOT));
    // ^ while we're here, object is definitely reachable, so not a root
    return std::make_tuple(l, h, m, n);
  } else {
    return std::make_tuple(MAX, 0, 0, 0);
  }
}
