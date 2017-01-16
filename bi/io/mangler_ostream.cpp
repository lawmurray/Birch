/**
 * @file
 */
#include "bi/io/mangler_ostream.hpp"

bi::mangler_ostream::mangler_ostream(std::ostream& base, const int level,
    const bool header) :
    bi_ostream(base, level, header) {
  //
}

void bi::mangler_ostream::visit(const ExpressionList* o) {
  if (dynamic_cast<VarParameter*>(o->head.get())) {
    *this << o->tail;
  } else {
    *this << o->head << ", " << o->tail;
  }
}

void bi::mangler_ostream::visit(const VarParameter* o) {
  //
}
