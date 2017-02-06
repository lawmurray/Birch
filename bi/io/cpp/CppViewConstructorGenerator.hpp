/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for model view constructors.
 *
 * @ingroup compiler_io
 */
class CppViewConstructorGenerator : public indentable_ostream {
public:
  CppViewConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const ModelParameter* o);

  void initialise(const VarParameter* o);
};
}
