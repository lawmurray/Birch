/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/assert.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
class Marker;
class Scanner;
class Reacher;
class Collector;
class Spanner;
class Bridger;
class Copier;

/**
 * Flags used for cycle collection as in @ref Bacon2001
 * "Bacon & Rajan (2001)", replacing the colors described there. The reason
 * is to ensure that both the bookkeeping required during normal execution
 * can be multithreaded, and likewise that the operations required during
 * cycle collection can be multithreaded. The basic principle to ensure this
 * is that flags can be safely set during normal execution (with atomic
 * operations), but should only be unset with careful consideration of
 * thread safety.
 *
 * Notwithstanding, the flags do map to colors in @ref Bacon2001
 * "Bacon & Rajan (2001)":
 *
 *   - *possible root* maps to *purple*,
 *   - *marked* maps to *gray*,
 *   - *scanned* and *reachable* together map to *black* (both on) or
 *     *white* (first on, second off),
 *   - *collected* is set once a white object has been destroyed.
 *
 * The use of these flags also resolves some thread safety issues that can
 * otherwise exist during the scan operation, when coloring an object white
 * (eligible for collection) then later recoloring it black (reachable); the
 * sequencing of this coloring can become problematic with multiple threads.
 */
enum Flag : uint8_t {
  POSSIBLE_ROOT = (1 << 0),
  BUFFERED = (1 << 1),
  MARKED = (1 << 2),
  SCANNED = (1 << 3),
  REACHED = (1 << 4),
  COLLECTED = (1 << 5),
  DESTROYED = (1 << 6),
  CLAIMED = (1 << 7)
};

/**
 * Base class providing reference counting, cycle breaking, and lazy deep
 * copy support.
 *
 * @ingroup libbirch
 *
 * @attention A newly created object of type Any, or of a type derived from
 * it, must be assigned to at least one Shared pointer in its lifetime to
 * be correctly destroyed and deallocated. Furthermore, in order to work
 * correctly with multiple inheritance, Any must be the *first* base class.
 */
class Any {
  friend class Marker;
  friend class Scanner;
  friend class Reacher;
  friend class Collector;
  friend class Spanner;
  friend class Bridger;
  friend class Copier;
public:
  using this_type_ = Any;

  /**
   * Constructor.
   */
  Any() :
      r(0),
      f(0),
      a(0),
      n(0),
      l(0),
      h(0),
      p(-1),
      allocTid(get_thread_num()) {
    //
  }

  /**
   * Copy constructor.
   */
  Any(const Any& o) : Any() {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Any() {
    assert(r.load() == 0);
  }

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    return allocate(size);
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    assert(false);
  }

  /**
   * Assignment operator.
   */
  Any& operator=(const Any&) {
    return *this;
  }

  /**
   * Destroy, but do not deallocate, the object.
   */
  void destroy() {
    assert(r.load() == 0);
    auto size = this->size_();
    auto tid = this->allocTid;
    this->~Any();
    libbirch::deallocate(this, size, tid);
  }

  /**
   * Shared r.
   */
  int numShared() const {
    return r.load();
  }

  /**
   * Increment the shared r.
   */
  void incShared() {
    //f.maskAnd(~POSSIBLE_ROOT);
    // ^ any interleaving with decShared() switching on POSSIBLE_ROOT should
    //   not be problematic; having it on is never a correctness issue, only
    //   a performance issue, and as long as one thread can reach the object
    //   it is fine to be off
    // ^ disabling this option improves performance on several examples
    r.increment();
  }

  /**
   * Decrement the shared r. This decrements the count; if the new count
   * is nonzero, it registers the object as a possible root for cycle
   * collection, or if the new count is zero, it destroys the object.
   */
  void decShared() {
    assert(numShared() > 0);

    /* if the count will reduce to nonzero, this is possibly the root of
     * a cycle */
    if (numShared() > 1 &&
        !(f.exchangeOr(BUFFERED|POSSIBLE_ROOT) & BUFFERED)) {
      register_possible_root(this);
    }

    /* decrement */
    if (--r == 0) {
      destroy();
    }
  }

  /**
   * Decrement the shared count with a known acyclic referent. This decrements
   * the count, and if the new count is zero, it destroys the object. The
   * caller asserts that the object is of acyclic type (@see is_acyclic), so
   * that there is no need to register the object as a possible root for cycle
   * collection.
   *
   * Acyclic objects occur in @ref Bacon2001 "Bacon & Rajan (2001)", where
   * they are colored *green*.
   */
  void decSharedAcyclic() {
    assert(numShared() > 0);
    if (--r == 0) {
      destroy();
    }
  }

  /**
   * Decrement the shared count for an object that will remain reachable. The
   * caller asserts that the object will remain reachable after the operation.
   * The object will not be destroyed, and will not be registered as a
   * possible root for cycle collection.
   */
  void decSharedReachable() {
    assert(numShared() > 0);
    r.decrement();
  }

  /**
   * Has the object been destroyed?
   */
  bool isDestroyed() const {
    return f.load() & DESTROYED;
  }

  /**
   * Is this object the possible root of a cycle?
   */
  bool isPossibleRoot() const {
    auto f = this->f.load();
    return (f & POSSIBLE_ROOT) && !(f & DESTROYED);
  }

  /**
   * Is there only one pointer (of any type) to this object?
   */
  bool isUnique() const {
    return numShared() == 1;
  }

  /**
   * Get the class name.
   */
  virtual const char* getClassName() const {
    return "Any";
  }

  /**
   * Size of the object.
   */
  int size() const {
    return size_();
  }
  virtual int size_() const = 0;

  /**
   * Shallow copy the object.
   */
  Any* copy() const {
    return copy_();
  }
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

private:
  /**
   * Reference count.
   */
  Atomic<int> r;

  /**
   * Bitfield containing flags, used for bridge finding and cycle collection.
   */
  Atomic<uint8_t> f;

  /**
   * Account of references, used for bridge finding and cycle collection.
   */
  int a:30;

  /**
   * Rank, used for bridge finding.
   */
  int n:30;

  /**
   * Lowest reachable rank, used for bridge finding.
   */
  int l:30;

  /**
   * Highest reachable rank, used for bridge finding.
   */
  int h:30;

  /**
   * Id of the thread that claimed the object, used for bridge finding.
   */
  int16_t p;

  /**
   * Id of the thread that allocated the object, used by the memory pool.
   */
  int16_t allocTid;
};
}
