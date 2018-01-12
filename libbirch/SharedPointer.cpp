/**
 * @file
 */
#include "libbirch/SharedPointer.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/World.hpp"
#include "libbirch/WeakPointer.hpp"

bi::SharedPointer<bi::Any>::SharedPointer() :
    SharedPointer(std::make_shared<Any>()) {
  //
}

bi::SharedPointer<bi::Any>::SharedPointer(const std::nullptr_t& o) {
  //
}

bi::SharedPointer<bi::Any>::SharedPointer(const std::shared_ptr<Any>& ptr) :
    ptr(ptr) {
  //
}

bi::SharedPointer<bi::Any>& bi::SharedPointer<bi::Any>::operator=(
    const std::nullptr_t& o) {
  ptr = o;
  return *this;
}

bool bi::SharedPointer<bi::Any>::query() const {
  return static_cast<bool>(ptr);
}

bi::Any* bi::SharedPointer<bi::Any>::get() const {
  /* copy-on-write */
  const_cast<std::shared_ptr<bi::Any>&>(ptr) = fiberWorld->get(ptr);
  return ptr.get();
}

bi::SharedPointer<const bi::Any>::SharedPointer() :
    SharedPointer(std::make_shared<const Any>()) {
  //
}

bi::SharedPointer<const bi::Any>::SharedPointer(const std::nullptr_t& o) {
  //
}

bi::SharedPointer<const bi::Any>::SharedPointer(
    const std::shared_ptr<const Any>& ptr) :
    ptr(ptr) {
  //
}

bi::SharedPointer<const bi::Any>::SharedPointer(
    const std::shared_ptr<Any>& ptr) :
    ptr(ptr) {
  //
}

bi::SharedPointer<const bi::Any>::SharedPointer(const SharedPointer<Any>& o) :
    ptr(o.ptr) {
  //
}

bi::SharedPointer<const bi::Any>& bi::SharedPointer<const bi::Any>::operator=(
    const std::nullptr_t& o) {
  ptr = o;
  return *this;
}

bi::SharedPointer<const bi::Any>& bi::SharedPointer<const bi::Any>::operator=(
    const SharedPointer<Any>& o) {
  ptr = o.ptr;
  return *this;
}

bool bi::SharedPointer<const bi::Any>::query() const {
  return static_cast<bool>(ptr);
}

const bi::Any* bi::SharedPointer<const bi::Any>::get() const {
  /* read-only, so no need for copy-on-write */
  return ptr.get();
}
