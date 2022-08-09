/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Does this operator exist in C++?
 */
bool isTranslatable(const std::string& op);

/**
 * Generate nice name. This replaces operators with words.
 */
std::string nice(const std::string& name);

/**
 * Generate C++ name. This is the original name, with any characters
 * outside the class [0-9a-zA-z_] translated to within that class, and an
 * underscore added to the end to avoid clashes with user variables.
 */
std::string sanitize(const std::string& name);

/**
 * Escape unicode characters in a string.
 */
std::string escape_unicode(const std::string& str);

/**
 * Convert a string to lower case.
 */
std::string lower(const std::string& str);

/**
 * Convert a string to upper case.
 */
std::string upper(const std::string& str);

/**
 * Tar name for a package.
 */
std::string tar(const std::string& name);

/**
 * Canonical name for a package. This is the same as the tar name, with
 * hyphens replaced with underscores.
 */
std::string canonical(const std::string& name);

}
