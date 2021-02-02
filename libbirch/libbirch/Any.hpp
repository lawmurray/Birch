/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/assert.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
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
 *   - *acyclic* maps to *green*,
 *   - *buffered* maps to *purple*,
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
  ACYCLIC = (1 << 0),
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
   * Destroy and deallocate.
   */
  void destroy() {
    auto size = this->size_();
    auto tid = this->allocTid;
    this->~Any();
    libbirch::deallocate(this, size, tid);
  }

  /**
   * Reference count.
   */
  int numShared() const {
    return r.load();
  }

  /**
   * Increment the shared r.
   */
  void incShared() {
    r.increment();
  }

  /**
   * Decrement the shared reference count.
   */
  void decShared() {
    assert(numShared() > 0);

    /* first set the BUFFERED flag so that this call is uniquely responsible
     * for registering this object as a possible root */
    auto old = f.exchangeOr(BUFFERED);

    if (--r == 0) {
      if ((old & ACYCLIC) || !(old & BUFFERED)) {
        /* either this object is acyclic and so doesn't need to be registered
         * as a possible root, or this call is uniquely responsible for
         * registering it as a possible root, but it cannot be; can just
         * destroy */
        destroy();
      }
    } else if (!(old & ACYCLIC) && !(old & BUFFERED)) {
      /* register as a possible root, but only if it's not acyclic */
      register_possible_root(this);
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
   * Is there only one pointer (of any type) to this object?
   */
  bool isUnique() const {
    return numShared() == 1;
  }

  /**
   * Is this object of an acyclic type?
   */
  bool isAcyclic() const {
    return f.load() & ACYCLIC;
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
