/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Flags used for bridge finding and cycle collection. For cycle collection,
 * they mostly correspond to the colors in @ref Bacon2001 "Bacon & Rajan
 * (2001)" but behave slightly differently to permit multithreading. The basic
 * principle to ensure this is that flags can be safely set during normal
 * execution (with atomic operations), but should only be unset with careful
 * consideration of thread safety.
 *
 * The flags map to colors in @ref Bacon2001 "Bacon & Rajan (2001)" as
 * follows:
 *
 *   - *buffered* maps to *purple*,
 *   - *marked* maps to *gray*,
 *   - *scanned* and *reached* together map to *black* (both on) or
 *     *white* (first on, second off),
 *   - *collected* is set once a *white* object has been destroyed.
 *
 * The use of these flags also resolves some thread safety issues that can
 * otherwise exist during the scan operation, when coloring an object white
 * (eligible for collection) then later recoloring it black (reachable); the
 * sequencing of this coloring can become problematic with multiple threads.
 * 
 * Acyclic objects are handled via separate mechanism, but map to *green*.
 */
enum Flag : int8_t {
  BUFFERED = (1 << 0),
  POSSIBLE_ROOT = (1 << 1),
  MARKED = (1 << 2),
  SCANNED = (1 << 3),
  REACHED = (1 << 4),
  COLLECTED = (1 << 5),
  CLAIMED = (1 << 6)
};

/**
 * Base class providing reference counting, cycle breaking, and lazy deep
 * copy support.
 *
 * @ingroup libbirch
 * 
 * Members of Any use an underscore suffix (e.g. `n_` instead of `n`) in order
 * to avoid naming collisions with derived classes.
 */
class Any {
  friend class Marker;
  friend class Scanner;
  friend class Reacher;
  friend class Collector;
  friend class BiconnectedCollector;
  friend class Spanner;
  friend class Bridger;
  friend class Copier;
  friend class Memo;
  friend class BiconnectedCopier;
  friend class BiconnectedMemo;
  friend class Destroyer;
public:
  using this_type_ = Any;

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
   * Assignment operator.
   */
  Any& operator=(const Any&);

  /**
   * Destroy.
   */
  void destroy_();

  /**
   * Deallocate.
   */
  void deallocate_();

  /**
   * Reference count.
   */
  int numShared_() const;

  /**
   * Increment the shared reference count.
   */
  void incShared_();

  /**
   * Decrement the shared reference count.
   */
  void decShared_();

  /**
   * Decrement the shared count for an object that will remain reachable. The
   * object will not be destroyed, and will not be registered as a possible
   * root for cycle collection.
   */
  void decSharedReachable_();

  /**
   * Decrement the shared count during collection of a biconnected component.
   */
  void decSharedBiconnected_();

  /**
   * Decrement the shared count for a bridge edge.
   */
  void decSharedBridge_();

  /**
   * Decrement the shared count for an acyclic edge.
   */
  void decSharedAcyclic_();

  /**
   * Is there only one reference to this object?
   */
  bool isUnique_() const;

  /**
   * Is there only one external reference to this object, assuming that it is
   * the head node of a biconnected component?
   */
  bool isUniqueHead_() const;

  /**
   * Is this object of an acyclic class?
   */
  virtual bool isAcyclic_() const;

  /**
   * Is this object the possible root of a cycle?
   */
  bool isPossibleRoot_() const;
  
  /**
   * Unset buffered flag.
   */
  void unbuffer_();

  /**
   * Get the class name.
   */
  virtual const char* getClassName_() const;

  /**
   * Shallow copy the object.
   */
  virtual Any* copy_() const = 0;

  virtual void accept_(Marker& visitor) {
    //
  }

  virtual void accept_(Scanner& visitor) {
    //
  }

  virtual void accept_(Reacher& visitor) {
    //
  }

  virtual void accept_(Collector& visitor) {
    //
  }

  virtual void accept_(BiconnectedCollector& visitor) {
    //
  }

  virtual std::tuple<int,int,int> accept_(Spanner& visitor, const int i,
      const int j) {
    return std::make_tuple(i, i, 0);
  }

  virtual std::tuple<int,int,int,int> accept_(Bridger& visitor, const int j,
      const int k) {
    return std::make_tuple(std::numeric_limits<int>::max(), 0, 0, 0);
  }

  virtual void accept_(Copier& visitor) {
    //
  }

  virtual void accept_(BiconnectedCopier& visitor) {
    //
  }

  virtual void accept_(Destroyer& visitor) {
    //
  }

private:
  /**
   * Reference count.
   */
  Atomic<int> r_;

  /**
   * Account of references, used for bridge finding. For the head of a
   * biconnected component (i.e. HEAD flag is set), this is the number of
   * internal references to the object, plus one for the bridge edge.
   */
  int a_;

  union {
    /**
     * Lowest reachable rank, used for bridge finding.
     */
    int l_;

    /**
     * Index base in biconnected component, used for copying.
     */
    int k_;
  };

  union {
    /**
     * Highest reachable rank, used for bridge finding.
     */
    int h_;

    /**
     * Size of biconnnected component, used for copying.
     */
    int n_;
  };

  /**
   * Id of the thread that claimed the object, used for bridge finding.
   */
  int16_t p_;

  /**
   * Bitfield containing flags, used for bridge finding and cycle collection.
   */
  Atomic<int8_t> f_;
};
}

inline libbirch::Any::~Any() {
  assert(r_.load() == 0);
}

inline libbirch::Any& libbirch::Any::operator=(const Any&) {
  return *this;
}

inline void libbirch::Any::deallocate_() {
  delete this;
}

inline int libbirch::Any::numShared_() const {
  return r_.load();
}

inline void libbirch::Any::incShared_() {
  r_.increment();
}

inline void libbirch::Any::decSharedReachable_() {
  assert(numShared_() > 0);
  r_.decrement();
}

inline bool libbirch::Any::isUnique_() const {
  return numShared_() == 1;
}

inline bool libbirch::Any::isUniqueHead_() const {
  return numShared_() == a_;
}

inline bool libbirch::Any::isAcyclic_() const {
  return false;
}

inline bool libbirch::Any::isPossibleRoot_() const {
  return f_.load() & POSSIBLE_ROOT;
}

inline void libbirch::Any::unbuffer_() {
  f_.maskAnd(~(BUFFERED|POSSIBLE_ROOT));
}

inline const char* libbirch::Any::getClassName_() const {
  return "Any";
}
