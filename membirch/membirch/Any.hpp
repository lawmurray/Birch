/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"
#include "membirch/internal.hpp"
#include "membirch/macro.hpp"
#include "membirch/memory.hpp"
#include "membirch/thread.hpp"
#include "membirch/Atomic.hpp"
#include "membirch/Marker.hpp"
#include "membirch/Scanner.hpp"
#include "membirch/Reacher.hpp"
#include "membirch/Collector.hpp"
#include "membirch/BiconnectedCollector.hpp"
#include "membirch/Spanner.hpp"
#include "membirch/Bridger.hpp"
#include "membirch/Copier.hpp"
#include "membirch/BiconnectedCopier.hpp"
#include "membirch/Destroyer.hpp"

namespace membirch {
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
  using base_type_ = void;
 
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
    return membirch::make_object<Any>(*this);
  }

  virtual void accept_(membirch::Marker& visitor_) {
    //
  }
 
  virtual void accept_(membirch::Scanner& visitor_) {
    //
  }
 
  virtual void accept_(membirch::Reacher& visitor_) {
    //
  }
 
  virtual void accept_(membirch::Collector& visitor_) {
    //
  }
 
  virtual void accept_(membirch::BiconnectedCollector& visitor_) {
    //
  }
 
  virtual std::tuple<int,int,int> accept_(membirch::Spanner& visitor_,
      const int i_, const int j_) {
    return visitor_.visit(i_, j_);
  }
 
  virtual std::tuple<int,int,int,int> accept_(membirch::Bridger& visitor_,
      const int j_, const int k_) {
    return visitor_.visit(j_, k_);
  }
 
  virtual void accept_(membirch::Copier& visitor_) {
    //
  }
 
  virtual void accept_(membirch::BiconnectedCopier& visitor_) {
    //
  }
 
  virtual void accept_(membirch::Destroyer& visitor_) {
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
