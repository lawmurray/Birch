/**
 * @file
 */
#include "bi/visitor/IsRich.hpp"

bi::IsRich::IsRich() : result(false) {
  //
}

void bi::IsRich::visit(const EmptyExpression* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const EmptyStatement* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const BoolLiteral* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const IntLiteral* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const RealLiteral* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const StringLiteral* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const Name* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const Path* o) {
  Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ExpressionList* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const StatementList* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const TypeList* o) {
  Visitor::visit(o);

}

void bi::IsRich::visit(const ParenthesesExpression* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const BracesExpression* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const BracketsExpression* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Range* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Member* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const This* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const RandomRight* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const VarReference* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const FuncReference* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const RandomReference* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const ModelReference* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const VarParameter* o) {
  Visitor::visit(o);
}

void bi::IsRich::visit(const FuncParameter* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const RandomParameter* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ModelParameter* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ProgParameter* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const File* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Import* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ExpressionStatement* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Conditional* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Loop* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const Raw* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const FuncDeclaration* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ModelDeclaration* o) {
  //Visitor::visit(o);
  result = true;
}

void bi::IsRich::visit(const ProgDeclaration* o) {
  //Visitor::visit(o);
  result = true;
}
