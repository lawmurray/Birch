/**
 * @file
 */
#include "src/common/Name.hpp"

#include "src/visitor/all.hpp"

int birch::Name::COUNTER = 0;

birch::Name::Name() {
  std::stringstream buf;
  buf << 'n' << (COUNTER++) << '_';
  name = buf.str();
}

birch::Name::Name(const std::string& name) :
    name(name) {
  //
}

birch::Name::Name(const char* name) :
    Name(std::string(name, strlen(name))) {
  //
}

birch::Name::Name(const char name) :
    Name(std::string(1, name)) {
  //
}

const std::string& birch::Name::str() const {
  return name;
}

bool birch::Name::isEmpty() const {
  return name.empty();
}

void birch::Name::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::Name::operator==(const Name& o) const {
  return name.compare(o.name) == 0;
}

bool birch::Name::operator!=(const Name& o) const {
  return !(*this == o);
}
