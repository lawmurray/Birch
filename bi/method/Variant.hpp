/**
 * @file
 */
#pragma once

#include "boost/variant.hpp"

namespace bi {
/**
 * Variant type.
 */
template<class ... Args>
using Variant = boost::variant<Args...>;
}
