/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/Based.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

#include <set>
#include <list>

namespace bi {
/**
 * Possible states of a class during parsing.
 */
enum ClassState {
  CLONED = 0, RESOLVED_SUPER = 1, RESOLVED_HEADER = 2, RESOLVED_SOURCE = 3
};

/**
 * Class.
 *
 * @ingroup birch_statement
 */
class Class: public Statement,
    public Named,
    public Numbered,
    public Parameterised,
    public Based,
    public Argumented,
    public Braced,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeParams Generic type parameters.
   * @param params Constructor parameters.
   * @param base Base type.
   * @param alias Is this an alias relationship?
   * @param args Base type constructor arguments.
   * @param braces Braces.
   * @param loc Location.
   */
  Class(Name* name, Expression* typeParams, Expression* params, Type* base,
      const bool alias, Expression* args, Statement* braces, Location* loc =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~Class();

  /**
   * Does this class have generic type parameters?
   */
  bool isGeneric() const;

  /**
   * Add a super type.
   */
  void addSuper(const Type* o);

  /**
   * Is the given type a super type of this?
   */
  bool hasSuper(const Type* o) const;

  /**
   * Add a conversion.
   */
  void addConversion(const Type* o);

  /**
   * Does this class have a conversion for the given type?
   */
  bool hasConversion(const Type* o) const;

  /**
   * Add an assignment.
   */
  void addAssignment(const Type* o);

  /**
   * Does this class have an assignment for the given type?
   */
  bool hasAssignment(const Type* o) const;

  /**
   * Add an instantiation.
   */
  void addInstantiation(Class* o);

  /**
   * Get the instantiation, if any, that exactly matches the given generic
   * type arguments. Returns `nullptr` if not such instantiation exists.
   */
  Class* getInstantiation(const Type* typeArgs);

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Generic type parameters.
   */
  Expression* typeParams;

  /**
   * State of the class during parsing. Classes with generic type parameters
   * are instantiated on use, which is why this is necessary: a new class
   * may be instantiated during a later parsing phase and we must keep track
   * of where we are for each class.
   */
  ClassState state;

  /**
   * Super classes.
   */
  std::set<const Class*> supers;

  /**
   * Types to which this class can be converted.
   */
  std::list<const Type*> conversions;

  /**
   * Types that can be assigned to this class.
   */
  std::list<const Type*> assignments;

  /**
   * Instantiations of this class.
   */
  std::list<Class*> instantiations;
};
}
