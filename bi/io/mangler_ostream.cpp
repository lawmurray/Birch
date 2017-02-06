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
  *this << o->head;
  if (o->head->isRich() && o->tail->isRich()) {
    *this << ", ";
  }
  *this << o->tail;
}

void bi::mangler_ostream::visit(const VarParameter* o) {
  if (o->isRich()) {
    *this << ':' << o->type;
  }
}
