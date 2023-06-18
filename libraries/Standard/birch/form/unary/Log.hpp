/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Log {
  BIRCH_UNARY_FORM(Log)
  BIRCH_UNARY_SIZE(Log)
  BIRCH_UNARY_EVAL(Log, log)
  BIRCH_UNARY_GRAD(Log, log_grad)
};

BIRCH_UNARY_TYPE(Log)
BIRCH_UNARY_CALL(Log, log)

}
