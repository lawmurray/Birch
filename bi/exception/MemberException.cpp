/**
 * @file
 */
#include "bi/exception/MemberException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::MemberException::MemberException(const Member* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: '.' used with non-class type on left\n";
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: in\n";
  buf << expr << '\n';
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: type on left is\n";
  buf << expr->left->type << '\n';
  msg = base.str();
}

bi::MemberException::MemberException(const MemberType* type) {
  std::stringstream base;
  bih_ostream buf(base);
  if (type->loc) {
    buf << type->loc;
  }
  buf << "error: '.' used with non-class type on left\n";
  if (type->loc) {
    buf << type->loc;
  }
  buf << "note: in\n";
  buf << type << '\n';
  if (type->loc) {
    buf << type->loc;
  }
  buf << "note: type on left is\n";
  buf << type->left << '\n';
  msg = base.str();
}
