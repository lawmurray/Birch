/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for multiple dispatchers.
 *
 * @ingroup compiler_io
 */
class CppDispatcherGenerator: public CppBaseGenerator {
public:
  CppDispatcherGenerator(Scope* scope, std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const File* o);
  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);

private:
  Scope* scope;
};
}
