/**
 * @file
 */
#pragma once

#include "bi/exception/FileNotFoundException.hpp"

#define STRINGIFY_IMPL(arg) #arg
#define STRINGIFY(arg) STRINGIFY_IMPL(arg)

namespace bi {
/**
 * Output a warning message.
 */
void warn(const std::string& msg);

/**
 * Find a path in a list of possible locations.
 */
fs::path find(const std::list<fs::path>& paths, const fs::path path);

/**
 * Copy a source file to a destination file, but only if the destination
 * file does not exist, or the source file is newer.
 *
 * @param src Source file.
 * @param dst Destination file.
 *
 * @return True if the file was copied, because of the above criteria.
 */
bool copy_if_newer(fs::path src, fs::path dst);

/**
 * Copy a source file to a destination file, but only overwrite an existing
 * file after prompting the user.
 *
 * @param src Source file.
 * @param dst Destination file.
 *
 * @return True if the file was copied, because of the above criteria.
 */
bool copy_with_prompt(fs::path src, fs::path dst);

/**
 * Copy a source file to a destination file, overwriting always.
 *
 * @param src Source file.
 * @param dst Destination file.
 */
void copy_with_force(fs::path src, fs::path dst);

/**
 * Remove the current directory (.) from the start of a path.
 */
fs::path remove_first(const fs::path& path);

/**
 * Remove the common prefix of both base and path from path, and return the
 * result.
 */
fs::path remove_common_prefix(const fs::path& base, const fs::path& path);

/**
 * Read the entirety of a file to a string.
 */
std::string read_all(const fs::path& path);

/**
 * Write the entirety of a file from a string.
 */
void write_all(const fs::path& path, const std::string& contents);

/**
 * Write the entirety of a file from a string, but only if the new contents
 * differs from the old contents.
 *
 * @return True if the contents differs, and so the file was written.
 */
bool write_all_if_different(const fs::path& path,
    const std::string& contents);

/**
 * Internal name for a package.
 */
std::string tarname(const std::string& name);

/**
 * Is an integer a positive power of two?
 */
bool isPower2(const int x);

/**
 * Change the working directory and restore it on destruction.
 */
class CWD {
public:
  CWD(const fs::path& path) : previous(fs::absolute(fs::current_path())) {
    fs::current_path(path);
  }

  ~CWD() {
    fs::current_path(previous);
  }

private:
  fs::path previous;
};

}
