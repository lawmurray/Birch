/**
 * @file
 */
#include "libbirch/Spanner.hpp"

#include "libbirch/Any.hpp"

std::tuple<int,int,int> libbirch::Spanner::visitObject(const int i,
    const int j, Any* o) {
  if (!(o->f_.exchangeOr(CLAIMED) & CLAIMED)) {
    /* just claimed by this thread */
    assert(o->p_ == -1);
    o->p_ = get_thread_num();
    o->a_ = 1;
    o->l_ = j;
    o->h_ = j;
    int l, h, k;
    std::tie(l, h, k) = o->accept_(*this, j, j + 1);
    o->l_ = std::min(o->l_, l);
    o->h_ = std::max(o->h_, h);
    return std::make_tuple(j, j, k + 1);
  } else if (o->p_ == get_thread_num()) {
    /* previously claimed by this thread */
    ++o->a_;
    o->l_ = std::min(o->l_, i);
    o->h_ = std::max(o->h_, i);
    return std::make_tuple(o->l_, o->h_, 0);
  } else {
    /* claimed by a different thread */
    return std::make_tuple(i, i, 0);
  }
}
