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

  virtual void visit(const File* o);
  virtual void visit(const Import* o);

  virtual void visit(const VarDeclaration* o);
  virtual void visit(const ModelParameter* o);
  virtual void visit(const ProgParameter* o);

protected:
  /**
   * Convert path to C++ header file name.
   */
  static std::string hpp(const std::string& path);
};
}
