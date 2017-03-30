/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for models.
 *
 * @ingroup compiler_io
 */
class CppModelGenerator: public CppBaseGenerator {
public:
  CppModelGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const ModelParameter* o);
  virtual void visit(const ModelReference* o);
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncParameter* o);

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
