/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for random variable components.
 *
 * @ingroup compiler_io
 */
class CppRandomGenerator : public indentable_ostream {
public:
  CppRandomGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const ModelParameter* o);
};
}
