/**
 * @file
 */
#include "libubjpp/value.hpp"

libubjpp::value::value() :
    value(object_type()) {
  //
}

boost::optional<libubjpp::value_type&> libubjpp::value::get(
    const std::initializer_list<std::string>& path) {
  auto node = this;
  for (auto name : path) {
    if (node->x.type() == typeid(object_type)) {
      auto& o = boost::get<object_type>(node->x);
      auto iter = o.find(name);
      if (iter != o.end()) {
        node = &iter->second;
      } else {
        return boost::none;
      }
    } else {
      return boost::none;
    }
  }
  return node->get();
}

boost::optional<const libubjpp::value_type&> libubjpp::value::get(
    const std::initializer_list<std::string>& path) const {
  auto node = this;
  for (auto name : path) {
    if (node->x.type() == typeid(object_type)) {
      auto& o = boost::get<object_type>(node->x);
      auto iter = o.find(name);
      if (iter != o.end()) {
        node = &iter->second;
      } else {
        return boost::none;
      }
    } else {
      return boost::none;
    }
  }
  return node->get();
}

boost::optional<libubjpp::value_type&> libubjpp::value::get(
    const std::string& name) {
  if (x.type() == typeid(object_type)) {
    auto& o = boost::get<object_type&>(x);
    auto iter = o.find(name);
    if (iter != o.end()) {
      return iter->second.get();
    }
  }
  return boost::none;
}

boost::optional<const libubjpp::value_type&> libubjpp::value::get(
    const std::string& name) const {
  if (x.type() == typeid(object_type)) {
    auto& o = boost::get<object_type&>(x);
    auto iter = o.find(name);
    if (iter != o.end()) {
      return iter->second.get();
    }
  }
  return boost::none;
}
