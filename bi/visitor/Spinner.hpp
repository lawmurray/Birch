/**
 * @file
 */
#pragma once

#include "bi/visitor/ContextualModifier.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Extract use of the spin operator into a preceding while loop that yields
 * until the fiber is complete.
 *
 * @ingroup visitor
 */
class Spinner: public ContextualModifier {
public:
  virtual Statement* modify(ExpressionStatement* o);
};
}
