/**
 * @file
 */
#pragma once

#include "bi/visitor/ScopedModifier.hpp"
#include "bi/visitor/Cloner.hpp"

namespace bi {
/**
 * Populate local scopes, and resolve identifiers.
 *
 * @ingroup visitor
 */
class Resolver: public ScopedModifier {
public:
  /**
   * Constructor.
   */
  Resolver();

  /**
   * Destructor.
   */
  virtual ~Resolver();

  using ScopedModifier::modify;

  virtual Expression* modify(Parameter* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Expression* modify(NamedExpression* o);
  virtual Type* modify(NamedType* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Yield* o);

protected:
  /**
   * Add parameters to a resume function.
   *
   * @param map Map containing parameters or local variables as values.
   * @param params Current parameters, `nullptr` if none.
   */
  template<class Map>
  Expression* addParameters(const Map& map, Expression* params = nullptr);

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
};
}

template<class Map>
bi::Expression* bi::Resolver::addParameters(const Map& map,
    Expression* params) {
  for (auto o : map) {
    auto p = o.second;
    auto param = new Parameter(NONE, p->name, p->type->accept(&cloner),
        new EmptyExpression(p->loc), p->loc);
    if (!params) {
      params = param;
    } else {
      params = new ExpressionList(param, params, param->loc);
    }
  }
  return params;
}
