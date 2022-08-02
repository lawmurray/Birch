/**
 * @file
 */
#include "libbirch/Reacher.hpp"

#include "libbirch/Any.hpp"

void libbirch::Reacher::visitObject(Any* o) {
  if (!(o->f_.exchangeOr(SCANNED) & SCANNED)) {
    o->f_.maskAnd(~MARKED);  // unset for next time
  }
  if (!(o->f_.exchangeOr(REACHED) & REACHED)) {
    o->accept_(*this);
  }
}
