/**
 * @file
 */
#include "membirch/Reacher.hpp"

#include "membirch/Any.hpp"

void membirch::Reacher::visitObject(Any* o) {
  if (!(o->f_.exchangeOr(SCANNED) & SCANNED)) {
    o->f_.maskAnd(~MARKED);  // unset for next time
  }
  if (!(o->f_.exchangeOr(REACHED) & REACHED)) {
    o->accept_(*this);
  }
}
