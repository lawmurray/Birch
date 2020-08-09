/**
 * @file
 */
#include "bi/common/Name.hpp"

#include "bi/visitor/all.hpp"

int bi::Name::COUNTER = 0;

bi::Name::Name() {
  std::stringstream buf;
  buf << 'n' << (COUNTER++) << '_';
  name = buf.str();
}

bi::Name::Name(const std::string& name) :
    name(name) {
  //
}

bi::Name::Name(const char* name) :
    Name(std::string(name, strlen(name))) {
  //
}

bi::Name::Name(const char name) :
    Name(std::string(1, name)) {
  //
}

bi::Name::~Name() {
  //
}

const std::string& bi::Name::str() const {
  return name;
}

bool bi::Name::isEmpty() const {
  return name.empty();
}

void bi::Name::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Name::operator==(const Name& o) const {
  return name.compare(o.name) == 0;
}

bool bi::Name::operator!=(const Name& o) const {
  return !(*this == o);
}
