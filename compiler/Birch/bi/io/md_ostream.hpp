/**
 * @file
 */
#pragma once

#include "bi/io/bih_ostream.hpp"

namespace bi {
/**
 * Output stream for Markdown files.
 *
 * @ingroup birch_io
 */
class md_ostream: public bih_ostream {
public:
  md_ostream(std::ostream& base);

  using bih_ostream::visit;

  virtual void visit(const Package* o);
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
  virtual void visit(const Basic* o);
  virtual void visit(const Class* o);

  virtual void visit(const TypeList* o);
  virtual void visit(const NamedType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const TupleType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const OptionalType* o);

private:
  void genHead(const std::string& name);

  /**
   * Package.
   */
  const Package* package;

  /**
   * Current section depth.
   */
  int depth;
};
}
