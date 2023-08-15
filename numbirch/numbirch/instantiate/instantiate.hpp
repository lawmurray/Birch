/**
 * @file
 */
#pragma once

#include <variant>

namespace numbirch {

using numeric_variant = std::variant<
    real,
    int,
    bool,
    Array<real,0>,
    Array<int,0>,
    Array<bool,0>,
    Array<real,1>,
    Array<int,1>,
    Array<bool,1>,
    Array<real,2>,
    Array<int,2>,
    Array<bool,2>
    >;

using arithmetic_variant = std::variant<
    real,
    int,
    bool
    >;

using scalar_variant = std::variant<
    real,
    int,
    bool,
    Array<real,0>,
    Array<int,0>,
    Array<bool,0>
    >;

using vector_variant = std::variant<
    Array<real,1>,
    Array<int,1>,
    Array<bool,1>
    >;

using matrix_variant = std::variant<
    Array<real,2>,
    Array<int,2>,
    Array<bool,2>
    >;

}
