/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/type/TypeIterator.hpp"
#include "bi/type/TypeConstIterator.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/all.hpp"

bi::Type::Type(Location* loc) :
    Located(loc) {
  //
}

bi::Type::~Type() {
  //
}

bool bi::Type::isEmpty() const {
  return false;
}

bool bi::Type::isValue() const {
  IsValue visitor;
  accept(&visitor);
  return visitor.result;
}

bool bi::Type::isBasic() const {
  return false;
}

bool bi::Type::isClass() const {
  return false;
}

bool bi::Type::isArray() const {
  return false;
}

bool bi::Type::isList() const {
  return false;
}

bool bi::Type::isFunction() const {
  return false;
}

bool bi::Type::isFiber() const {
  return false;
}

bool bi::Type::isMember() const {
  return false;
}

bool bi::Type::isOptional() const {
  return false;
}

bool bi::Type::isWeak() const {
  return false;
}

bool bi::Type::isGeneric() const {
  return false;
}

int bi::Type::width() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    ++result;
  }
  return result;
}

int bi::Type::depth() const {
  return 0;
}

bi::Type* bi::Type::unwrap() {
  return this;
}

const bi::Type* bi::Type::unwrap() const {
  return this;
}

bi::Type* bi::Type::canonical() {
  return this;
}

const bi::Type* bi::Type::canonical() const {
  return this;
}

bi::Type* bi::Type::element() {
  return this;
}

const bi::Type* bi::Type::element() const {
  return this;
}

bi::TypeIterator bi::Type::begin() {
  if (isEmpty()) {
    return end();
  } else {
    return TypeIterator(this);
  }
}

bi::TypeConstIterator bi::Type::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return TypeConstIterator(this);
  }
}

bi::TypeIterator bi::Type::end() {
  return TypeIterator(nullptr);
}

bi::TypeConstIterator bi::Type::end() const {
  return TypeConstIterator(nullptr);
}
