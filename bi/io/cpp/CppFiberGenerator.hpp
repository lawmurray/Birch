/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

namespace bi {
/**
 * C++ code generator for fibers.
 *
 * @ingroup io
 */
class CppFiberGenerator: public CppBaseGenerator {
public:
  CppFiberGenerator(std::ostream& base,
      const int level = 0, const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Fiber* o);
  virtual void visit(const Return* o);
  virtual void visit(const Yield* o);
  virtual void visit(const Identifier<Parameter>* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const For* o);

protected:
  /**
   * Get the unique name for a local variable.
   */
  std::string getName(const std::string& name, const int number);

  /**
   * The yield type of the fiber being generated.
   */
  const Type* yieldType;

  /*
   * Gatherers for important objects.
   */
  Gatherer<Parameter> params;
  Gatherer<LocalVariable> locals;
  Gatherer<Yield> yields;

  /**
   * Name mappings. Local variables become member variables of the fiber
   * state object, where they cannot be distinguished by scope, and must
   * therefore have unique names. This map assigns unique names by simply
   * appending a count to each subsequent occurrence of the same name. The
   * names remain simple and predictable, which is of benefit when nesting
   * C++ code within Birch functions. The map is indexed by the unique number
   * of each variable.
   */
  std::map<int,std::string> names;

  /**
   * Counts of all local variable names encountered.
   */
  std::map<std::string,int> counts;

  /**
   * Current yield point.
   */
  int point;

  /**
   * Are we in a for loop?
   */
  bool inFor;
};
}
