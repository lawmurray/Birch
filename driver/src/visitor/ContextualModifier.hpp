/**
 * @file
 */
#pragma once

#include "src/visitor/Modifier.hpp"
#include "src/expression/all.hpp"
#include "src/statement/all.hpp"
#include "src/type/all.hpp"

namespace birch {
/**
 * Modifier that keeps track of the current context (e.g. within a package,
 * class or fiber).
 *
 * @ingroup visitor
 */
class ContextualModifier: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param currentPackage Initial package context.
   * @param currentClass Initial class context.
   * @param currentFiber Initial fiber context.
   */
  ContextualModifier(Package* currentPackage = nullptr,
      Class* currentClass = nullptr, Fiber* currentFiber = nullptr);

  /**
   * Destructor.
   */
  virtual ~ContextualModifier();

  using Modifier::modify;

  virtual Package* modify(Package* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(MemberFiber* o);

protected:
  /**
   * If in a package, a pointer to that package, otherwise `nullptr`.
   */
  Package* currentPackage;

  /**
   * If in a class, a pointer to that class, otherwise `nullptr`.
   */
  Class* currentClass;

  /**
   * If in a fiber, a pointer to that fiber, otherwise `nullptr`.
   */
  Fiber* currentFiber;
};
}
