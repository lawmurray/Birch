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
  CppDispatcherGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const File* o);
  virtual void visit(const Dispatcher* o);
  virtual void visit(const VarParameter* o);

  void genBody(const Dispatcher* o);
  void genArg(const Expression* arg, const VarParameter* param);
};
}
