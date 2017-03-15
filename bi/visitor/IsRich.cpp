/**
 * @file
 */
#include "bi/visitor/IsRich.hpp"

bi::IsRich::IsRich() : result(false) {
  //
}

bi::IsRich::~IsRich() {
  //
}

void bi::IsRich::visit(const BracesExpression* o) {
  result = true;
}

void bi::IsRich::visit(const BracketsExpression* o) {
  result = true;
}

void bi::IsRich::visit(const Index* o) {
  result = true;
}

void bi::IsRich::visit(const Range* o) {
  result = true;
}

void bi::IsRich::visit(const Member* o) {
  result = true;
}

void bi::IsRich::visit(const This* o) {
  result = true;
}

void bi::IsRich::visit(const LambdaInit* o) {
  result = true;
}

void bi::IsRich::visit(const RandomInit* o) {
  result = true;
}

void bi::IsRich::visit(const VarReference* o) {
  result = true;
}

void bi::IsRich::visit(const FuncReference* o) {
  result = true;
}
