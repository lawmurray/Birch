/**
 * @file
 */
#pragma once

#include "bi/io/bih_ostream.hpp"
#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"

#include <list>

namespace bi {
/**
 * Output stream for Markdown files.
 *
 * @ingroup compiler_io
 */
class md_ostream: public bih_ostream {
public:
  md_ostream(std::ostream& base, const std::list<File*> files);

  using bih_ostream::visit;

  void gen();

  virtual void visit(const Class* o);

private:
  void genHead(const std::string& name);

  template<class ObjectType>
  void genSection();

  template<class ObjectType>
  void genClassSection(const Class* o);

  /**
   * Files to include in documentation.
   */
  std::list<File*> files;

  /**
   * Current section depth.
   */
  int depth;
};
}

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

template<class ObjectType>
void bi::md_ostream::genSection() {
  Gatherer<ObjectType> gatherer;
  for (auto file : files) {
    file->accept(&gatherer);
  }

  ++depth;
  for (auto o : gatherer) {
    auto named = dynamic_cast<Named*>(o);
    if (named) {
      genHead(named->name->str());
    }
    if (!o->loc->doc.empty()) {
      line(comment(o->loc->doc));
    }
    in();
    in();
    line(o);
    out();
    out();
    line("\n");
  }
  --depth;
}

template<class ObjectType>
void bi::md_ostream::genClassSection(const Class* o) {
  Gatherer<ObjectType> gatherer;
  o->accept(&gatherer);

  ++depth;
  for (auto o : gatherer) {
    auto named = dynamic_cast<Named*>(o);
    if (named) {
      genHead(named->name->str());
    }
    if (!o->loc->doc.empty()) {
      line(comment(o->loc->doc));
    }
    in();
    in();
    line(o);
    out();
    out();
    line("\n");
  }
  --depth;
}
