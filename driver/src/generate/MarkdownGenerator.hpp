/**
 * @file
 */
#pragma once

#include "src/generate/BirchGenerator.hpp"

namespace birch {
/**
 * Output stream for Markdown files.
 *
 * @ingroup birch_io
 */
class MarkdownGenerator: public BirchGenerator {
public:
  MarkdownGenerator(std::ostream& base);

  using BirchGenerator::visit;

  virtual void visit(const Package* o);
  virtual void visit(const Name* o);

  virtual void visit(const Parameter* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const Program* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const SliceOperator* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Struct* o);
  virtual void visit(const Class* o);

  virtual void visit(const TypeList* o);
  virtual void visit(const NamedType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const TupleType* o);
  virtual void visit(const OptionalType* o);
  virtual void visit(const DeducedType* o);

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
