/**
 * @file
 */
#pragma once

#include "bi/statement/all.hpp"
#include "bi/exception/all.hpp"

#include "boost/filesystem.hpp"

#include <unordered_set>
#include <unordered_map>

namespace bi {
/**
 * Birch compiler.
 *
 * @ingroup compiler
 */
class Compiler {
public:
  /**
   * Default constructor.
   */
  Compiler();

  /**
   * Constructor from command-line options.
   */
  Compiler(int argc, char** argv);

  /**
   * Destructor.
   */
  virtual ~Compiler();

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
   * Import a path.
   *
   * @param path Path given in import statement.
   *
   * @return File associated with the path.
   */
  File* import(const Path* path);

  /**
   * Queue a file for parsing.
   *
   * @param name File name.
   * @param std Include the standard library automatically?
   *
   * @return File associated with the path.
   */
  File* queue(const std::string name, const bool std = false);

  /**
   * Set the root statement of the file, adding an import for the
   * standard library if requested.
   */
  void setRoot(Statement* root);

  /**
   * Current file being parsed (needed by GNU Bison parser).
   */
  File* file;

  /**
   * All files.
   */
  std::list<File*> files;

  /**
   * Standard library inclusion flag of current file being parsed.
   */
  bool std;

private:
  /**
   * Parse a specific file.
   *
   * @param name File name.
   */
  void parse(const std::string name);

  /**
   * File names to files.
   */
  std::unordered_map<std::string,File*> filesByName;

  /**
   * File names to standard library inclusion flags.
   */
  std::unordered_map<std::string,bool> stds;

  /**
   * Unparsed files.
   */
  std::unordered_set<std::string> unparsed;

  /**
   * Parsed files.
   */
  std::unordered_set<std::string> parsed;

  /**
   * @name Command-line options
   */
  //@{
  /**
   * Input file.
   */
  boost::filesystem::path input_file;

  /**
   * Output file.
   */
  boost::filesystem::path output_file;

  /**
   * Include directories.
   */
  std::list<boost::filesystem::path> include_dirs;

  /**
   * Library directories.
   */
  std::list<boost::filesystem::path> lib_dirs;

  /**
   * Name of standard library.
   */
  std::string standard;

  /**
   * Enable standard library?
   */
  bool enable_std;

  /**
   * Enable imports? (Can be disabled when no resolution is to happen, such
   * as when generating reference documentation.)
   */
  bool enable_import;
  //@}
};
}

extern bi::Compiler* compiler;
extern std::stringstream raw;
