/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for models.
 *
 * @ingroup compiler_io
 */
class CppModelGenerator: public indentable_ostream {
public:
  CppModelGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const Name* o);
  virtual void visit(const ModelParameter* o);
  virtual void visit(const ModelReference* o);
  virtual void visit(const BracketsType* o);

  virtual void visit(const VarDeclaration* o);
  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);

  virtual void visit(const Raw* o);

protected:
  /**
   * The model being generated.
   */
  const ModelParameter* model;

  /**
   * Currently generating type for an array?
   */
  bool inArray;
};
}
