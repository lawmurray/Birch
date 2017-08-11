/**
 * @file
 */
#pragma once

#include "bi/io/bih_ostream.hpp"

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

  virtual void visit(const Name* o);

  virtual void visit(const Parameter* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const Fiber* o);
  virtual void visit(const Program* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const MemberFiber* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const Class* o);

  virtual void visit(const ListType* o);
  virtual void visit(const ClassType* o);
  virtual void visit(const BasicType* o);
  virtual void visit(const AliasType* o);
  virtual void visit(const IdentifierType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const OptionalType* o);

private:
  void genHead(const std::string& name);

  void genClasses(const std::string& name);

  template<class ObjectType>
  void genBrief(const std::string& name);

  template<class ObjectType>
  void genBrief(const std::string& name, const Class* o);

  template<class ObjectType>
  void genDetailed(const std::string& name);

  template<class ObjectType>
  void genDetailed(const std::string& name, const Class* o);

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
void bi::md_ostream::genBrief(const std::string& name) {
  Gatherer<ObjectType> gatherer;
  for (auto file : files) {
    file->accept(&gatherer);
  }
  if (gatherer.size() > 0) {
    genHead(name);
    line("| --- | --- |");
    ++depth;
    for (auto o : gatherer) {
      line("| *" << o << "* | " << brief(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }
}

template<class ObjectType>
void bi::md_ostream::genBrief(const std::string& name, const Class* o) {
  Gatherer<ObjectType> gatherer;
  o->accept(&gatherer);
  if (gatherer.size() > 0) {
    genHead(name);
    line("| --- | --- |");
    ++depth;
    for (auto o : gatherer) {
      line("| *" << o << "* | " << brief(o->loc->doc) << " |");
    }
    line("");
    --depth;
  }
}

template<class ObjectType>
void bi::md_ostream::genDetailed(const std::string& name) {
  Gatherer<ObjectType> gatherer;
  for (auto file : files) {
    file->accept(&gatherer);
  }
  if (gatherer.size() > 0) {
    genHead(name);
    ++depth;
    for (auto o : gatherer) {
      std::string desc = detailed(o->loc->doc);
      line("| --- |");
      line("| *" << o << "* |");
      if (!desc.empty()) {
        line("| " << desc << " |");
      }
      line("");
    }
    --depth;
  }
}

template<class ObjectType>
void bi::md_ostream::genDetailed(const std::string& name, const Class* o) {
  Gatherer<ObjectType> gatherer;
  o->accept(&gatherer);
  if (gatherer.size() > 0) {
    genHead(name);
    ++depth;
    for (auto o : gatherer) {
      std::string desc = detailed(o->loc->doc);
      line("| --- |");
      line("| *" << o << "* |");
      if (!desc.empty()) {
        line("| " << desc << " |");
      }
      line("");
    }
    --depth;
  }
}
