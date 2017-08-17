/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"

namespace bi {
/**
 * Visitor to import declarations from imported scopes.
 *
 * @ingroup compiler_visitor
 */
class Importer : public Modifier {
public:
  /**
   * Destructor.
   */
  virtual ~Importer();

  /**
   * Perform one iteration of imports for a file.
   *
   * @param file The file.
   *
   * @return Were any new imports made?
   */
  bool import(File* file);

  using Modifier::modify;
  virtual Statement* modify(Import* o);

private:
  /**
   * The file.
   */
  File* file;

  /**
   * Have any new imports been discovered in this iteration?
   */
  bool haveNew;
};
}
