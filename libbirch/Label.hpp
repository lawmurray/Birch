/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Label for bookkeeping lazy deep clones.
 *
 * @ingroup libbirch
 */
class Label final: public Any {
public:
  LIBBIRCH_CLASS(Label, Any)
  LIBBIRCH_MEMBERS(memo)

  /**
   * Constructor.
   */
  Label();

  /**
   * Copy constructor.
   */
  Label(const Label& o);

  /**
   * Destructor.
   */
  virtual ~Label() {
    //
  }

  /**
   * Update a pointer for writing.
   *
   * @param Pointer type (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  void get(P& o)  {
    if (o.query() && o->isFrozen()) {
      lock.write();
      Any* old = o.get();
      Any* ptr = get(old);
      if (ptr != old) {
        o.replace(static_cast<typename P::value_type*>(ptr));
      }
      lock.unwrite();
    }
  }

  /**
   * Update a pointer for reading.
   *
   * @param Pointer type (SharedPtr, WeakPtr or InitPtr).
   */
  template<class P>
  void pull(P& o) {
    if (o.query() && o->isFrozen()) {
      lock.read();
      Any* old = o.get();
      Any* ptr = pull(old);
      lock.unread();
      if (ptr != old) {
        /* it is possible for multiple threads to try to update o
         * simultaneously, and the interleaving operations to result in
         * incorrect reference counts updates; ensure exclusive access with a
         * write lock */
        lock.write();
        o.replace(static_cast<typename P::value_type*>(ptr));
        lock.unwrite();
      }
    }
  }

private:
  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  Any* get(Any* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  Any* pull(Any* o);

  /**
   * Memo that maps source objects to clones.
   */
  Memo memo;

  /**
   * Lock.
   */
  ReaderWriterLock lock;
};
}

namespace bi {
  namespace type {
template<>
struct super_type<libbirch::Label> {
  using type = libbirch::Any;
};
  }
}
