/**
 * @file
 */
#pragma once

#include <string>

namespace bi {
/**
 * Does this operator exist in C++?
 */
bool isTranslatable(const std::string& op);

/**
 * Translate an operator to C++.
 */
std::string translate(const std::string& op);
}
