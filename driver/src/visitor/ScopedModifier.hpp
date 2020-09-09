/**
 * @file
 */
#pragma once

#include "src/visitor/ContextualModifier.hpp"
#include "src/expression/all.hpp"
#include "src/statement/all.hpp"
#include "src/type/all.hpp"

namespace birch {
/**
 * Modifier that keeps track of the stack of scopes during traversal.
 *
 * @ingroup visitor
 */
class ScopedModifier: public ContextualModifier {
public:
  /**
   * Constructor.
   *
   * @param currentPackage If the visitor will not begin by visiting the
   * package, provide it for scoping purposes.
   * @param currentClass If the visitor will begin by visiting the members of
   * a class, but not the class itself, provide it for scoping purposes.
   */
  ScopedModifier(Package* currentPackage = nullptr,
      Class* currentClass = nullptr);

  /**
   * Destructor.
   */
  virtual ~ScopedModifier();

  using Modifier::modify;

  virtual Package* modify(Package* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Global* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(Parallel* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(DoWhile* o);
  virtual Statement* modify(With* o);
  virtual Statement* modify(Block* o);

protected:
  /**
   * List of scopes, innermost at the back.
   */
  std::list<Scope*> scopes;

  /**
   * Are we on the right hand side of a member dereference (i.e. `b` in
   * `a.b`)?
   */
  int inMember;

  /**
   * Are we on the right hand side of a global dereference (i.e. `b` in
   * `global.b`)?
   */
  int inGlobal;
};
}
