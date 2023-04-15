/**
 * @file
 */
#include "src/statement/Package.hpp"

#include "src/visitor/all.hpp"

birch::Package::Package(const std::string& name,
    const std::list<File*>& sources) :
    name(name),
    sources(sources) {
  //
}

void birch::Package::addPackage(const std::string& name) {
  packages.push_back(name);
}

void birch::Package::addSource(const std::string& path) {
  sources.push_back(new File(path));
}

void birch::Package::accept(Visitor* visitor) const {
  visitor->visit(this);
}
