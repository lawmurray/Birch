/**
 * @file
 */
#include "libbirch/WeakPointer.hpp"

#include "libbirch/SharedPointer.hpp"
#include "libbirch/Any.hpp"

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

bi::Optional<bi::SharedPointer<bi::Any>> bi::WeakPointer<bi::Any>::lock() const {
  return bi::Optional<SharedPointer<Any>>(ptr.lock());
}
