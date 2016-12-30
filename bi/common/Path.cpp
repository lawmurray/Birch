/**
 * @file
 */
#include "bi/common/Path.hpp"

#include "bi/visitor/all.hpp"

#include "boost/filesystem.hpp"

#include <sstream>

bi::Path::Path(shared_ptr<Name> head, Path* tail, shared_ptr<Location> loc) :
    Located(loc), head(head), tail(tail) {
  //
}

inline bi::Path::~Path() {
  //
}

void bi::Path::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Path::accept(Visitor* visitor) const {
  visitor->visit(this);
}


bool bi::Path::operator<=(Path& o) {
  return *this == o;
}

bool bi::Path::operator==(const Path& o) const {
  return *head == *o.head
      && ((!tail && !o.tail) || (tail && o.tail && *tail == *o.tail));
}

inline bool bi::Path::operator<(Path& o) {
  return *this <= o && *this != o;
}

inline bool bi::Path::operator>(Path& o) {
  return !(*this <= o);
}

inline bool bi::Path::operator>=(Path& o) {
  return !(*this < o);
}

inline bool bi::Path::operator!=(Path& o) {
  return !(*this == o);
}

std::string bi::Path::file() const {
  boost::filesystem::path file("bi");
  const Path* path = this;
  while (path) {
    file /= path->head->str();
    path = path->tail.get();
  }
  file.replace_extension("bi");

  return file.string();
}

std::string bi::Path::str() const {
  std::stringstream buf;
  const Path* path = this;
  while (path) {
    buf << path->head->str();
    path = path->tail.get();
    if (path) {
      buf << '.';
    }
  }
  return buf.str();
}
