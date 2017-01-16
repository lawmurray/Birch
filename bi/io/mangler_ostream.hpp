/**
 * @file
 */
#pragma once

#include "bi/io/bi_ostream.hpp"

namespace bi {
/**
 * Output stream for mangling identifiers for use in C++. Removes the names
 * and types of parameters, as C++ can resolve overloads without these.
 *
 * @ingroup compiler_io
 */
class mangler_ostream: public bi_ostream {
public:
  mangler_ostream(std::ostream& base, const int level = 0, const bool header =
      false);

  virtual void visit(const ExpressionList* o);
  virtual void visit(const VarParameter* o);
};
}
