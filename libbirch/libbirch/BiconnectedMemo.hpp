/**
 * @file
 */
#pragma once

#include "libbirch/ReadersWriterLock.hpp"

namespace libbirch {
class Any;

/**
 * Memo for copying graphs of known size, such as for biconnected components,
 * implemented as an array indexed by the sequential ranks assigned to
 * vertices during bridge finding.
 *
 * @ingroup libbirch
 */
class BiconnectedMemo {
public:
  /**
   * Constructor.
   * 
   * @param o The bridge head.
   */
  BiconnectedMemo(Any* o);

  /**
   * Destructor.
   */
  ~BiconnectedMemo();

  /**
   * Get reference to the value associated with a key, which may be `nullptr`,
   * in which case the value may be written.
   *
   * @param key Key.
   */
  Any*& get(Any* key);

private:
  /**
   * The values.
   */
  Any** values;

  /**
   * Offset of ranks in the biconnected component.
   */
  int offset;

  /**
   * Number of entries in the table.
   */
  int nentries;
};
}
