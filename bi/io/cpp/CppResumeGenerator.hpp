/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

namespace bi {
/**
 * C++ code generator for resume functions of fibers.
 *
 * @ingroup io
 */
class CppResumeGenerator: public CppBaseGenerator {
public:
  CppResumeGenerator(const Class* currentClass, const Fiber* currentFiber,
      std::ostream& base, const int level = 0, const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Function* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const Yield* o);
  virtual void visit(const Return* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const NamedExpression* o);

protected:
  /**
   * Get a unique name for a local variable. This incorporates the unique id
   * number of the variable into its name, so as to separate local variables
   * of the same name declared in differently-scoped blocks.
   */
  std::string getName(const std::string& name, const int number);

  /**
   * Generate the name of a loop index.
   */
  virtual std::string getIndex(const Statement* o);

  /**
   * Generate a unique name for a resume function.
   *
   * @param o A (resume) Function or Yield.
   */
  void genUniqueName(const Numbered* o);

  /**
   * Generate code for type of pack.
   */
  void genPackType(const Function* o);

  /**
   * Generate code for packing parameters into a tuple.
   */
  void genPackParam(const Function* o);

  /**
   * Generate code for packing local variables into a tuple.
   */
  void genPackLocal(const Function* o);

  /**
   * Generate code for unpacking parameters from a tuple.
   */
  void genUnpackParam(const Function* o);

  /**
   * Generate code for the macro used to yield at a particular yield point.
   */
  void genYieldMacro(const Function* o);

  /**
   * Does the resume function require any parameters to be preserved?
   */
  bool requiresParam(const Function* o);

  /**
   * Does the resume function require any local variables to be preserved?
   */
  bool requiresLocal(const Function* o);

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
   * The class.
   */
  const Class* currentClass;

  /**
   * The fiber.
   */
  const Fiber* currentFiber;

  /**
   * Index of next parameter when unpacking.
   */
  int paramIndex;

  /**
   * Index of next local variable when unpacking.
   */
  int localIndex;
};
}
