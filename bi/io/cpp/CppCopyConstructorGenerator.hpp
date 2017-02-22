/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for model copy constructors.
 *
 * @ingroup compiler_io
 */
class CppCopyConstructorGenerator : public indentable_ostream {
public:
  CppCopyConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const ModelParameter* o);

  void initialise(const VarParameter* o);
};
}
