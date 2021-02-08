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
  POSSIBLE_ROOT = (1 << 0),
  BUFFERED = (1 << 1),
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
  friend class Memo;
  friend class BiconnectedCopier;
  friend class BiconnectedMemo;
public:
  using this_type_ = Any;

  /**
   * Constructor.
   */
  Any() :
      r(0),
      a(0),
      l(std::numeric_limits<int>::max()),
      h(0),
      k(0),
      n(0),
      p(-1),
      allocTid(get_thread_num()),
      f(0) {
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
   * Destroy.
   */
  void destroy() {
    auto tid = this->allocTid;
    auto size = this->size_();
    this->~Any();
    this->allocTid = tid;
    this->n = size;
  }

  /**
   * Deallocate.
   */
  void deallocate() {
    libbirch::deallocate(this, this->n, this->allocTid);
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
    auto old = f.exchangeOr(BUFFERED|POSSIBLE_ROOT);

    if (--r == 0) {
      /* destroy, and as long as haven't been previously buffered, can
       * deallocate too */
      destroy();
      if (!(old & BUFFERED)) {
        /* hasn't been previously buffered, so can immediately deallocate */
        deallocate();
      } else {
        /* has been previously buffered, so deallocation must be deferred
         * until collection, but certainly not a possible root, as has just
         * been destroyed */
        f.maskAnd(~POSSIBLE_ROOT);
      }
    } else if (!(old & BUFFERED)) {
      /* register as a possible root, as not already */
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
   * Is this object the possible root of a cycle?
   */
  bool isPossibleRoot() const {
    return f.load() & POSSIBLE_ROOT;
  }

  /**
   * Unset buffer flag.
   */
  void unbuffer() {
    f.maskAnd(~(BUFFERED|POSSIBLE_ROOT));
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
  virtual int size_() const {
    return sizeof(Any);
  }

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

  virtual void accept_(BiconnectedCopier& visitor) {
    //
  }

private:
  /**
   * Reference count.
   */
  Atomic<int> r;

  /**
   * Account of references, used for bridge finding and cycle collection.
   */
  int a;

  /**
   * Lowest reachable rank, used for bridge finding.
   */
  int l;

  /**
   * Highest reachable rank, used for bridge finding.
   */
  int h;

  /**
   * Index base in biconnected component, used for copying.
   */
  int k;

  /**
   * Size of biconnnected component, used for copying.
   */
  int n;

  /**
   * Id of the thread that claimed the object, used for bridge finding.
   */
  int16_t p;

  /**
   * Id of the thread that allocated the object, used by the memory pool.
   */
  int16_t allocTid;

  /**
   * Bitfield containing flags, used for bridge finding and cycle collection.
   */
  Atomic<uint8_t> f;
};
}
