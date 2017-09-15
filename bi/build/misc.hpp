/**
 * @file
 */
#pragma once

#include "bi/exception/FileNotFoundException.hpp"

#include "boost/filesystem.hpp"

#include <list>

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
boost::filesystem::path find(const std::list<boost::filesystem::path>& paths,
    const boost::filesystem::path path);

/**
 * Copy a source file to a destination file, but only if the destination
 * file does not exist, or the source file is newer.
 *
 * @param src Source file.
 * @param dst Destination file.
 *
 * @return True if the file was copied, because of the above criteria.
 */
bool copy_if_newer(boost::filesystem::path src, boost::filesystem::path dst);

/**
 * Copy a source file to a destination file, but only overwrite an existing
 * file after prompting the user.
 *
 * @param src Source file.
 * @param dst Destination file.
 *
 * @return True if the file was copied, because of the above criteria.
 */
bool copy_with_prompt(boost::filesystem::path src,
    boost::filesystem::path dst);

/**
 * Copy a source file to a destination file, overwriting always.
 *
 * @param src Source file.
 * @param dst Destination file.
 */
void copy_with_force(boost::filesystem::path src,
    boost::filesystem::path dst);

/**
 * Remove the current directory (.) from the start of a path.
 */
boost::filesystem::path remove_first(const boost::filesystem::path& path);

/**
 * Read the entirety of a file to a string.
 */
std::string read_all(const boost::filesystem::path& path);

/**
 * Write the entirety of a file from a string.
 */
void write_all(const boost::filesystem::path& path,
    const std::string& contents);

/**
 * Write the entirety of a file from a string, but only if the new contents
 * differs from the old contents.
 */
void write_all_if_different(const boost::filesystem::path& path,
    const std::string& contents);

}
