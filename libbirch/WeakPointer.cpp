/**
 * @file
 */
#include "libbirch/WeakPointer.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/World.hpp"
#include "libbirch/SharedPointer.hpp"

bi::WeakPointer<bi::Any>::WeakPointer(const std::nullptr_t& o) {
  //
}

bi::WeakPointer<bi::Any>::WeakPointer(const SharedPointer<Any>& o) :
    ptr(o.ptr) {
  //
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  ptr.reset();
  return *this;
}

bi::WeakPointer<bi::Any>& bi::WeakPointer<bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  ptr = o.ptr;
  return *this;
}

bi::SharedPointer<bi::Any> bi::WeakPointer<bi::Any>::lock() const {
  return SharedPointer<Any>(ptr.lock());
}

bi::WeakPointer<const bi::Any>::WeakPointer(const std::nullptr_t& o) {
  //
}

bi::WeakPointer<const bi::Any>::WeakPointer(const SharedPointer<const Any>& o) :
    ptr(o.ptr) {
  //
}

bi::WeakPointer<const bi::Any>::WeakPointer(const WeakPointer<Any>& o) :
    ptr(o.ptr) {
  //
}

bi::WeakPointer<const bi::Any>::WeakPointer(const SharedPointer<Any>& o) :
    ptr(o.ptr) {
  //
}

bi::WeakPointer<const bi::Any>& bi::WeakPointer<const bi::Any>::operator=(
    const std::nullptr_t& o) {
  ptr.reset();
  return *this;
}

bi::WeakPointer<const bi::Any>& bi::WeakPointer<const bi::Any>::operator=(
    const SharedPointer<const Any>& o) {
  ptr = o.ptr;
  return *this;
}

bi::WeakPointer<const bi::Any>& bi::WeakPointer<const bi::Any>::operator=(
    const WeakPointer<Any>& o) {
  ptr = o.ptr;
  return *this;
}

bi::WeakPointer<const bi::Any>& bi::WeakPointer<const bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  ptr = o.ptr;
  return *this;
}

bi::SharedPointer<const bi::Any> bi::WeakPointer<const bi::Any>::lock() const {
  return SharedPointer<const Any>(ptr.lock());
}
