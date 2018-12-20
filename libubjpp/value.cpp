/**
 * @file
 */
#include "libubjpp/value.hpp"

libubjpp::value::value() :
    value(object_type()) {
  //
}

boost::optional<libubjpp::value&> libubjpp::value::get(
    const std::string& name) {
  if (x.type() == typeid(object_type)) {
    auto& o = boost::get<object_type>(x);
    auto iter = o.find(name);
    if (iter != o.end()) {
      return iter->second.get();
    }
  }
  return boost::none;
}

boost::optional<const libubjpp::value&> libubjpp::value::get(
    const std::string& name) const {
  if (x.type() == typeid(object_type)) {
    auto& o = boost::get<object_type>(x);
    auto iter = o.find(name);
    if (iter != o.end()) {
      return iter->second.get();
    }
  }
  return boost::none;
}

libubjpp::value& libubjpp::value::set(const std::string& name,
    const value_type& x) {
  assert(this->x.type() == typeid(object_type));
  return boost::get<object_type>(this->x)[name].set(x);
}

libubjpp::value& libubjpp::value::set(const value_type& x) {
  this->x = x;
  return *this;
}
