/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/macro.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Reacher.hpp"
#include "libbirch/Collector.hpp"
#include "libbirch/BiconnectedCollector.hpp"
#include "libbirch/Spanner.hpp"
#include "libbirch/Bridger.hpp"
#include "libbirch/Copier.hpp"
#include "libbirch/BiconnectedCopier.hpp"
#include "libbirch/Destroyer.hpp"

namespace libbirch {
/**
 * Base class providing reference counting, cycle breaking, and lazy deep
 * copy support.
 * 
 * Members of Any use an underscore suffix (e.g. `n_` instead of `n`) in order
 * to avoid naming collisions with derived classes.
 */
class Any {
public:
  using this_type_ = Any;
  using base_type_ = LIBBIRCH_NO_BASE;
 
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
   * @internal
   * 
   * Destroy.
   */
  void destroy_();

  /**
   * @internal
   * 
   * Deallocate.
   */
  void deallocate_();

  /**
   * @internal
   * 
   * Reference count.
   */
  int numShared_() const;

  /**
   * @internal
   * 
   * Increment the shared reference count.
   */
  void incShared_();

  /**
   * @internal
   * 
   * Decrement the shared reference count.
   */
  void decShared_();

  /**
   * @internal
   * 
   * Decrement the shared count for an object that will remain reachable. The
   * object will not be destroyed, and will not be registered as a possible
   * root for cycle collection.
   */
  void decSharedReachable_();

  /**
   * @internal
   * 
   * Decrement the shared count during collection of a biconnected component.
   */
  void decSharedBiconnected_();

  /**
   * @internal
   * 
   * Decrement the shared count for a bridge edge.
   */
  void decSharedBridge_();

  /**
   * @internal
   * 
   * Decrement the shared count for an acyclic edge.
   */
  void decSharedAcyclic_();

  /**
   * @internal
   * 
   * Is there only one reference to this object?
   */
  bool isUnique_() const;

  /**
   * @internal
   * 
   * Is there only one external reference to this object, assuming that it is
   * the head node of a biconnected component?
   */
  bool isUniqueHead_() const;

  /**
   * @internal
   * 
   * Is this object of an acyclic class?
   */
  virtual bool isAcyclic_() const;

  /**
   * @internal
   * 
   * Is this object the possible root of a cycle?
   */
  bool isPossibleRoot_() const;
  
  /**
   * @internal
   * 
   * Unset buffered flag.
   */
  void unbuffer_();

  virtual const char* getClassName_() const {
    return "Any";
  }
 
  virtual Any* copy_() const {
    return libbirch::make_object<Any>(*this);
  }

  virtual void accept_(libbirch::Marker& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::Scanner& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::Reacher& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::Collector& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::BiconnectedCollector& visitor_) {
    //
  }
 
  virtual std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_,
      const int i_, const int j_) {
    return visitor_.visit(i_, j_);
  }
 
  virtual std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_,
      const int j_, const int k_) {
    return visitor_.visit(j_, k_);
  }
 
  virtual void accept_(libbirch::Copier& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::BiconnectedCopier& visitor_) {
    //
  }
 
  virtual void accept_(libbirch::Destroyer& visitor_) {
    //
  }

private:
  /**
   * @internal
   * 
   * Reference count.
   */
  Atomic<int> r_;

  /**
   * @internal
   * 
   * Account of references, used for bridge finding. For the head of a
   * biconnected component (i.e. HEAD flag is set), this is the number of
   * internal references to the object, plus one for the bridge edge.
   */
  int a_;

  union {
    /**
     * @internal
     * 
     * Lowest reachable rank, used for bridge finding.
     */
    int l_;

    /**
     * @internal
     * 
     * Index base in biconnected component, used for copying.
     */
    int k_;
  };

  union {
    /**
     * @internal
     * 
     * Highest reachable rank, used for bridge finding.
     */
    int h_;

    /**
     * @internal
     * 
     * Size of biconnnected component, used for copying.
     */
    int n_;
  };

  /**
   * @internal
   * 
   * Id of the thread that claimed the object, used for bridge finding.
   */
  int16_t p_;

  /**
   * @internal
   * 
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
