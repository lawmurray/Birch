/**
 * @file
 */
#pragma once

#include "bi/statement/all.hpp"
#include "bi/exception/all.hpp"

#include "boost/filesystem.hpp"

#include <unordered_set>
#include <unordered_map>
#include <list>

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
  Compiler(const std::list<boost::filesystem::path>& include_dirs,
      const std::list<boost::filesystem::path> lib_dirs,
      const bool std = false);

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
   *
   * @return File associated with the path.
   */
  File* queue(const std::string name);

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
  bool std;
  //@}
};
}

extern bi::Compiler* compiler;
extern std::stringstream raw;
