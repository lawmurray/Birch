/**
 * @file
 */
#include "bi/common/Name.hpp"

#include "bi/visitor/all.hpp"

bi::Name::Name(const std::string& name, shared_ptr<Location> loc) :
    Located(loc), name(name) {
  //
}

bi::Name::Name(const char* name, shared_ptr<Location> loc) :
    Name(std::string(name, strlen(name)), loc) {
  //
}

bi::Name::Name(const char name, shared_ptr<Location> loc) :
    Name(std::string(1, name), loc) {
  //
}

bi::Name::Name(const char sigil, const std::string& name,
    shared_ptr<Location> loc) :
    Name(std::string(1, sigil) + name, loc) {
  //
}

bi::Name::Name(const char sigil, const char* name, shared_ptr<Location> loc) :
    Name(std::string(1, sigil) + std::string(name, strlen(name)), loc) {
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
