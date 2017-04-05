/**
 * @file
 */
#pragma once

#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/expression/FuncParameter.hpp"

#include <list>

namespace bi {
/**
 * Dispatcher for runtime resolution of a function call.
 *
 * @ingroup compiler_dispatcher
 */
class Dispatcher: public Named,
    public Numbered,
    public Parenthesised,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   */
  Dispatcher(shared_ptr<Name> name);

  /**
   * Destructor.
   */
  virtual ~Dispatcher();

  /**
   * Add a function.
   */
  void add(FuncParameter* o);

  virtual Dispatcher* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /*
   * Comparison operators.
   */
  bool operator==(const Dispatcher& o) const;

  /**
   * Functions in this dispatcher.
   */
  std::list<FuncParameter*> funcs;

  /**
   * Parameter types.
   */
  std::list<VariantType*> types;

  /**
   * Result type.
   */
  bi::unique_ptr<Type> type;
};
}
