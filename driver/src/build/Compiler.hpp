/**
 * @file
 */
#pragma once

#include "src/build/misc.hpp"
#include "src/statement/all.hpp"
#include "src/exception/all.hpp"

namespace birch {
/**
 * Compiler.
 *
 * @ingroup driver
 */
class Compiler {
public:
  /**
   * Constructor.
   *
   * @param package The package.
   * @param unit Compilation unit.
   */
  Compiler(Package* package, const std::string& unit);

  /**
   * Parse source files.
   * 
   * @param includeHeaders Include header files also?
   */
  void parse(bool includeHeaders);

  /**
   * Resolve identifiers and check types.
   */
  void resolve();

  /**
   * Generate output code for all input files.
   */
  void gen();

  /**
   * Set the root statement of the current file.
   */
  void setRoot(Statement* root);

  /**
   * Current file being parsed (needed by GNU Bison parser).
   */
  File* file;

  /**
   * Root scope.
   */
  Scope* scope;

private:
  /**
   * Package.
   */
  Package* package;

  /**
   * Compilation unit.
   */
  std::string unit;
};
}

extern birch::Compiler* compiler;
extern std::stringstream raw;
