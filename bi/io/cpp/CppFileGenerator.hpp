/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for files.
 *
 * @ingroup compiler_io
 */
class CppFileGenerator: public CppBaseGenerator {
public:
  CppFileGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const File* o);
  virtual void visit(const Import* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const Function* o);
  virtual void visit(const Coroutine* o);
  virtual void visit(const Program* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const TypeParameter* o);

protected:
  /**
   * Convert path to C++ header file name.
   */
  static std::string hpp(const std::string& path);
};
}
