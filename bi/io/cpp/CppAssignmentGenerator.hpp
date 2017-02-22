/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for model assignment operators.
 *
 * @ingroup compiler_io
 */
class CppAssignmentGenerator : public indentable_ostream {
public:
  CppAssignmentGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;
  virtual void visit(const ModelParameter* o);

  virtual void assign(const VarParameter* o);
};
}
