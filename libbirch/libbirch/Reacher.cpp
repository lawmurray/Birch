/**
 * @file
 */
#include "libbirch/Reacher.hpp"

void libbirch::Reacher::visit(Any* o) {
  if (!(o->f.exchangeOr(SCANNED) & SCANNED)) {
    o->f.maskAnd(~MARKED);  // unset for next time
  }
  if (!(o->f.exchangeOr(REACHED) & REACHED)) {
    o->accept_(*this);
  }
}
