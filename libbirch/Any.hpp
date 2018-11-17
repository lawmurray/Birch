/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/Map.hpp"

namespace bi {
template<class T> class SharedCOW;

/**
 * Base for all class types.
 *
 * @ingroup libbirch
 */
class Any: public Counted {
public:
  /**
   * Constructor.
   */
  Any();

  /**
   * Copy constructor.
   */
  Any(const Any& o);

  /**
   * Destructor.
   */
  virtual ~Any();

  /**
   * Clone the object.
   */
  virtual Any* clone(Memo* memo) const;

  /**
   * Deallocate the memory for the object.
   */
  virtual void destroy();

  /**
   * Get the memo associated with the clone of this object.
   */
  Memo* getMemo();

  /**
   * Set the memo associated with the clone of this object.
   */
  void setMemo(Memo* memo);

  /**
   * Shallow retrieval of an object that may not yet have been cloned,
   * cloning it if necessary.
   *
   * @param memo Memo associated with the clone.
   *
   * @return The mapped object.
   */
  Any* get(Memo* memo);

  /**
   * Shallow retrieval of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   *
   * @param memo Memo associated with the clone.
   *
   * @return The mapped object.
   */
  Any* pull(Memo* memo);

  /**
   * Deep retrieval of an object that may not yet have been cloned,
   * cloning it if necessary.
   *
   * @param memo Memo associated with the clone.
   *
   * @return The mapped object.
   */
  Any* deepGet(Memo* memo);

  /**
   * Deep retrieval of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   *
   * @param memo Memo associated with the clone.
   *
   * @return The mapped object.
   */
  Any* deepPull(Memo* memo);

protected:
  /**
   * Memo associated with the clone or construction of this object.
   */
  SharedPtr<Memo> memo;

  /**
   * Clones produced from this object.
   */
  Map clones;
};
}
