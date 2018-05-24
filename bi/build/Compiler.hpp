/**
 * @file
 */
#pragma once

#include "bi/build/misc.hpp"
#include "bi/statement/all.hpp"
#include "bi/exception/all.hpp"

#include <list>
#include <string>

namespace bi {
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
   * @param build_dir Build directory.
   * @param unity Use unity build?
   */
  Compiler(Package* package, const fs::path& build_dir,
      const bool unity = false);

  /**
   * Parse all input files.
   */
  void parse();

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
   * Build directory.
   */
  fs::path build_dir;

  /**
   * Use unity build?
   */
  bool unity;
};
}

extern bi::Compiler* compiler;
extern std::stringstream raw;
