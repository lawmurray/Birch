/**
 * @file
 */
#pragma once

#include "bi/statement/all.hpp"
#include "bi/exception/all.hpp"

#include "boost/filesystem.hpp"

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
   * @param work_dir Working directory.
   * @param build_dir Build directory.
   */
  Compiler(Package* package, const boost::filesystem::path& work_dir,
      const boost::filesystem::path& build_dir);

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
   * Working directory.
   */
  boost::filesystem::path work_dir;

  /**
   * Build directory.
   */
  boost::filesystem::path build_dir;
};
}

extern bi::Compiler* compiler;
extern std::stringstream raw;
