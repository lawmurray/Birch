/**
 * @file
 */
#include "bi/visitor/TupleSizer.hpp"

bi::TupleSizer::TupleSizer() : size(0), dims(0) {
  //
}

void bi::TupleSizer::visit(const EmptyExpression* o) {
  //
}

void bi::TupleSizer::visit(const BoolLiteral* o) {
  ++size;
}

void bi::TupleSizer::visit(const IntLiteral* o) {
  ++size;
}

void bi::TupleSizer::visit(const RealLiteral* o) {
  ++size;
}

void bi::TupleSizer::visit(const StringLiteral* o) {
  ++size;
}

void bi::TupleSizer::visit(const ExpressionList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::TupleSizer::visit(const ParenthesesExpression* o) {
  ++size;
}

void bi::TupleSizer::visit(const BracesExpression* o) {
  ++size;
}

void bi::TupleSizer::visit(const BracketsExpression* o) {
  ++size;
}

void bi::TupleSizer::visit(const Range* o) {
  ++size;
  ++dims;
}

void bi::TupleSizer::visit(const Traversal* o) {
  ++size;
}

void bi::TupleSizer::visit(const This* o) {
  ++size;
}

void bi::TupleSizer::visit(const VarReference* o) {
  ++size;
}

void bi::TupleSizer::visit(const FuncReference* o) {
  ++size;
}

void bi::TupleSizer::visit(const RandomReference* o) {
  ++size;
}

void bi::TupleSizer::visit(const ProgReference* o) {
  ++size;
}

void bi::TupleSizer::visit(const VarParameter* o) {
  ++size;
}

void bi::TupleSizer::visit(const FuncParameter* o) {
  ++size;
}

void bi::TupleSizer::visit(const RandomParameter* o) {
  ++size;
}

void bi::TupleSizer::visit(const ProgParameter* o) {
  ++size;
}
