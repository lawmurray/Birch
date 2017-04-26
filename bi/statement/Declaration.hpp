/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/TypeParameter.hpp"
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

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const Declaration<T>& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const Declaration<T>& o) const;
};

typedef Declaration<VarParameter> VarDeclaration;
typedef Declaration<FuncParameter> FuncDeclaration;
typedef Declaration<ProgParameter> ProgDeclaration;
typedef Declaration<TypeParameter> TypeDeclaration;
}
