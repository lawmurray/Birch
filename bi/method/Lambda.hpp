/**
 * @file
 */
#pragma once

#include <functional>

namespace bi {
/**
 * Lambda function.
 */
template<class T>
using Lambda = std::function<T>;
}
