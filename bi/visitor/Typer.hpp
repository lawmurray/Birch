/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"

namespace bi {
/**
 * This is the first pass of the abstract syntax tree after parsing,
 * populating available types. This is analogous to having forward
 * declarations of all types as in C++.
 *
 * @ingroup compiler_visitor
 */
class Typer: public Modifier {
public:
  /**
   * Constructor.
   */
  Typer();

  /**
   * Destructor.
   */
  virtual ~Typer();

  using Modifier::modify;

  virtual File* modify(File* file);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Alias* o);

protected:
  /**
   * The file.
   */
  File* file;
};
}
