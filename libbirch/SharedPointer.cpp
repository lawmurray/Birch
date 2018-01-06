/**
 * @file
 */
#include "libbirch/SharedPointer.hpp"

#include "libbirch/World.hpp"
#include "libbirch/WeakPointer.hpp"
#include "libbirch/Any.hpp"

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

bi::Any* bi::SharedPointer<bi::Any>::get() {
  return fiberWorld->get(ptr).get();
}

bi::Any* bi::SharedPointer<bi::Any>::get() const {
  return fiberWorld->get(ptr).get();
}
