/**
 * @file
 */
#include "bi/statement/Package.hpp"

#include "bi/visitor/all.hpp"

bi::Package::Package(const std::string& name, const std::list<File*>& headers,
    const std::list<File*>& sources) :
    Scoped(GLOBAL_SCOPE),
    name(name),
    headers(headers),
    sources(sources) {
  files.insert(files.end(), headers.begin(), headers.end());
  files.insert(files.end(), sources.begin(), sources.end());
}

bi::Package::~Package() {
  //
}

void bi::Package::addPackage(const std::string& name) {
  packages.push_back(name);
}

void bi::Package::addHeader(const std::string& path) {
  headers.push_back(new File(path));
  files.push_back(headers.back());
}

void bi::Package::addSource(const std::string& path) {
  sources.push_back(new File(path));
  files.push_back(sources.back());
}

bi::Package* bi::Package::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Package* bi::Package::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Package::accept(Visitor* visitor) const {
  visitor->visit(this);
}
