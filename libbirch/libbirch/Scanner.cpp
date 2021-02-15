/**
 * @file
 */
#include "libbirch/Scanner.hpp"

#include "libbirch/Reacher.hpp"

void libbirch::Scanner::visit(Any* o) {
  if (!(o->f_.exchangeOr(SCANNED) & SCANNED)) {
    o->f_.maskAnd(~MARKED);  // unset for next time
    if (o->numShared_() > 0) {
      if (!(o->f_.exchangeOr(REACHED) & REACHED)) {
        Reacher visitor;
        o->accept_(visitor);
      }
    } else {
      o->accept_(*this);
    }
  }
}
