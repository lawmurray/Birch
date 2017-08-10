/**
 * @file
 */
#pragma once

#include "bi/io/bih_ostream.hpp"

namespace bi {
/**
 * Output stream for Markdown files.
 *
 * @ingroup compiler_io
 */
class md_ostream: public bih_ostream {
public:
  md_ostream(std::ostream& base);

  using bih_ostream::visit;

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
  virtual void visit(const Alias* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Import* o);

private:
  void genDoc(const Location* o);
};
}
