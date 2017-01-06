/**
 * @file
 */
#include "bi/visitor/IsPrimary.hpp"

bi::IsPrimary::IsPrimary() : result(true) {
  //
}

void bi::IsPrimary::visit(const EmptyExpression* o) {
  result = true;
}

void bi::IsPrimary::visit(const EmptyStatement* o) {
  result = true;
}

void bi::IsPrimary::visit(const BoolLiteral* o) {
  result = true;
}

void bi::IsPrimary::visit(const IntLiteral* o) {
  result = true;
}

void bi::IsPrimary::visit(const RealLiteral* o) {
  result = true;
}

void bi::IsPrimary::visit(const StringLiteral* o) {
  result = true;
}

void bi::IsPrimary::visit(const Name* o) {
  result = true;
}

void bi::IsPrimary::visit(const Path* o) {
  result = true;
}

void bi::IsPrimary::visit(const ExpressionList* o) {
  result = false;
}

void bi::IsPrimary::visit(const StatementList* o) {
  result = false;
}

void bi::IsPrimary::visit(const TypeList* o) {
  result = false;
}

void bi::IsPrimary::visit(const ParenthesesExpression* o) {
  result = true;
}

void bi::IsPrimary::visit(const BracesExpression* o) {
  result = true;
}

void bi::IsPrimary::visit(const BracketsExpression* o) {
  result = false;
}

void bi::IsPrimary::visit(const Range* o) {
  result = false;
}

void bi::IsPrimary::visit(const Member* o) {
  result = false;
}

void bi::IsPrimary::visit(const This* o) {
  result = true;
}

void bi::IsPrimary::visit(const VarReference* o) {
  result = true;
}

void bi::IsPrimary::visit(const FuncReference* o) {
  result = true;
}

void bi::IsPrimary::visit(const RandomReference* o) {
  result = true;
}

void bi::IsPrimary::visit(const ModelReference* o) {
  result = false;
}

void bi::IsPrimary::visit(const VarParameter* o) {
  result = true;
}

void bi::IsPrimary::visit(const FuncParameter* o) {
  result = true;
}

void bi::IsPrimary::visit(const RandomParameter* o) {
  result = false;
}

void bi::IsPrimary::visit(const ModelParameter* o) {
  result = false;
}

void bi::IsPrimary::visit(const ProgParameter* o) {
  result = false;
}

void bi::IsPrimary::visit(const File* o) {
  result = false;
}

void bi::IsPrimary::visit(const Import* o) {
  result = false;
}

void bi::IsPrimary::visit(const ExpressionStatement* o) {
  result = false;
}

void bi::IsPrimary::visit(const Conditional* o) {
  result = false;
}

void bi::IsPrimary::visit(const Loop* o) {
  result = false;
}

void bi::IsPrimary::visit(const Raw* o) {
  result = false;
}

void bi::IsPrimary::visit(const VarDeclaration* o) {
  result = false;
}

void bi::IsPrimary::visit(const FuncDeclaration* o) {
  result = false;
}

void bi::IsPrimary::visit(const ModelDeclaration* o) {
  result = false;
}

void bi::IsPrimary::visit(const ProgDeclaration* o) {
  result = false;
}
