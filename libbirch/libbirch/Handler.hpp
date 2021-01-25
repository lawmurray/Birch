/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Any.hpp"

namespace libbirch {
/**
 * Base class for event handlers.
 *
 * @ingroup libbirch
 */
class Handler : public Any {
public:
  LIBBIRCH_CLASS(Handler, Any)
  LIBBIRCH_MEMBERS()

};
}
