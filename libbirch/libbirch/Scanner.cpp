/**
 * @file
 */
#include "libbirch/Scanner.hpp"

#include "libbirch/Reacher.hpp"

void libbirch::Scanner::visit(Any* o) {
  if (!(o->f.exchangeOr(SCANNED) & SCANNED)) {
    o->f.maskAnd(~MARKED);  // unset for next time
    if (o->numShared() > 0) {
      if (!(o->f.exchangeOr(REACHED) & REACHED)) {
        Reacher visitor;
        o->accept_(visitor);
      }
    } else {
      o->accept_(*this);
    }
  }
}
