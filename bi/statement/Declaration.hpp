/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/ModelParameter.hpp"
#include "bi/program/ProgParameter.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Declaration.
 *
 * @ingroup compiler_statement
 */
template<class T>
class Declaration: public Statement {
public:
  /**
   * Constructor.
   *
   * @param param The parameter being declared.
   * @param loc Location.
   */
  Declaration(T* param, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Declaration();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Parameter.
   */
  unique_ptr<T> param;

  virtual bool dispatch(Statement& o);
  virtual bool le(Declaration<T>& o);
};

typedef Declaration<VarParameter> VarDeclaration;
typedef Declaration<FuncParameter> FuncDeclaration;
typedef Declaration<ProgParameter> ProgDeclaration;
typedef Declaration<ModelParameter> ModelDeclaration;
}
