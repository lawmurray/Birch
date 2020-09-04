/**
 * @file
 */
#include "src/statement/Package.hpp"

#include "src/visitor/all.hpp"

birch::Package::Package(const std::string& name, const std::list<File*>& headers,
    const std::list<File*>& sources) :
    Scoped(GLOBAL_SCOPE),
    name(name),
    headers(headers),
    sources(sources) {
  files.insert(files.end(), headers.begin(), headers.end());
  files.insert(files.end(), sources.begin(), sources.end());
}

birch::Package::~Package() {
  //
}

void birch::Package::addPackage(const std::string& name) {
  packages.push_back(name);
}

void birch::Package::addHeader(const std::string& path) {
  headers.push_back(new File(path));
  files.push_back(headers.back());
}

void birch::Package::addSource(const std::string& path) {
  sources.push_back(new File(path));
  files.push_back(sources.back());
}

birch::Package* birch::Package::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Package* birch::Package::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Package::accept(Visitor* visitor) const {
  visitor->visit(this);
}
