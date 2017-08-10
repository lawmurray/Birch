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
   * Constructor.
   */
  Compiler(int argc, char** argv);

  /**
   * Destructor.
   */
  ~Compiler();

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
   * Generate documentation for all input files.
   */
  void doc();

  /**
   * Import a path.
   *
   * @param path Path given in import statement.
   *
   * @return File associated with the path.
   */
  File* import(const Path* path);

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
   * Standard library inclusion flag of current file being parsed.
   */
  bool std;

private:
  /**
   * Queue a file for parsing.
   *
   * @param name File name.
   * @param std Include the standard library automatically?
   *
   * @return File associated with the path.
   */
  File* queue(const std::string name, const bool std);

  /**
   * Parse a specific file.
   *
   * @param name File name.
   */
  void parse(const std::string name);

  /**
   * Generate output code for specific file.
   *
   * @param name File name.
   */
  void gen(const std::string name);

  /**
   * Generate documentation for specific file.
   *
   * @param name File name.
   */
  void doc(const std::string name);

  /**
   * Set state of all files.
   */
  void setStates(const File::State state);

  /**
   * File names to files.
   */
  std::unordered_map<std::string,File*> files;

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
   * Files for which to generate code.
   */
  std::list<std::string> input_files;

  /**
   * Name of standard library.
   */
  std::string standard;

  /**
   * Enable standard library?
   */
  bool enable_std;
  //@}
};
}
