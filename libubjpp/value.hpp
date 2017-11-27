#pragma once

//#define BOOST_VARIANT_MINIMIZE_SIZE 1
#include "boost/variant.hpp"
#include "boost/optional.hpp"

#include <map>
#include <list>
#include <string>

namespace libubjpp {
class value;

/**
 * Object type. Note std::map is preferred to std::unordered_map due to its
 * size: 24 bytes as opposed to 32 bytes, which makes each value object 32
 * bytes as opposed to 48 bytes.
 */
using object_type = std::map<std::string,value>;

/**
 * Array type.
 */
using array_type = std::list<value>;

/**
 * String type.
 */
using string_type = std::string;

/*
 * Floating point types.
 */
using float_type = float;
using double_type = double;

/*
 * Integer types.
 */
using int8_type = int8_t;
using uint8_type = uint8_t;
using int16_type = int16_t;
using int32_type = int32_t;
using int64_type = int64_t;

/**
 * Boolean type.
 */
using bool_type = bool;

/**
 * Nil type.
 */
struct nil_type {
  //
};

/**
 * No-op type.
 */
struct noop_type {
  //
};

/**
 * Variant value type.
 */
using value_type = boost::variant<object_type,array_type,string_type,
float_type,double_type,int8_type,uint8_type,int16_type,int32_type,
int64_type,bool_type,nil_type,noop_type>;

/**
 * Value.
 *
 * @ingroup libubjpp
 */
class value {
public:
  /**
   * Default constructor. Creates a value of object type.
   */
  value();

  /**
   * Constructor.
   *
   * @param T value type.
   *
   * @param value x.
   */
  template<class T>
  value(const T& x) :
      x(x) {
    //
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @param path List of strings giving names of the items.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<T&> get(const std::initializer_list<std::string>& path) {
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
    return node->get<T>();
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @param path List of strings giving names of the items.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<const T&> get(
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
    return node->get<T>();
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @param name Name of the item.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<T&> get(const std::string& name) {
    if (x.type() == typeid(object_type)) {
      auto& o = boost::get<object_type&>(x);
      auto iter = o.find(name);
      if (iter != o.end()) {
        return iter->second.get<T>();
      }
    }
    return boost::none;
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @param name Name of the item.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<const T&> get(const std::string& name) const {
    if (x.type() == typeid(object_type)) {
      auto& o = boost::get<object_type&>(x);
      auto iter = o.find(name);
      if (iter != o.end()) {
        return iter->second.get<T>();
      }
    }
    return boost::none;
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<T&> get() {
    if (x.type() == typeid(T)) {
      return boost::get<T>(x);
    } else {
      return boost::none;
    }
  }

  /**
   * Get.
   *
   * @tparam T value type.
   *
   * @return An optional that will be empty if the value does not exist or
   * is not of type @p T, otherwise it will contain the x.
   */
  template<class T>
  boost::optional<const T&> get() const {
    if (x.type() == typeid(T)) {
      return boost::get<T>(x);
    } else {
      return boost::none;
    }
  }

  /**
   * Get.
   *
   * @param path List of strings giving names of the items.
   *
   * @return An optional that will be empty if the value does not exist,
   * otherwise it will contain the value of variant type.
   */
  boost::optional<value_type&> get(
      const std::initializer_list<std::string>& path);

  /**
   * Get.
   *
   * @param path List of strings giving names of the items.
   *
   * @return An optional that will be empty if the value does not exist,
   * otherwise it will contain the value of variant type.
   */
  boost::optional<const value_type&> get(
      const std::initializer_list<std::string>& path) const;

  /**
   * Get.
   *
   * @param name Name of the item.
   *
   * @return An optional that will be empty if the value does not exist,
   * otherwise it will contain the value of variant type.
   */
  boost::optional<value_type&> get(const std::string& name);

  /**
   * Get.
   *
   * @param name Name of the item.
   *
   * @return An optional that will be empty if the value does not exist,
   * otherwise it will contain the value of variant type.
   */
  boost::optional<const value_type&> get(const std::string& name) const;

  /**
   * Get.
   *
   * @return The value, of variant type.
   */
  value_type& get() {
    return x;
  }

  /**
   * Get.
   *
   * @return The value, of variant type.
   */
  const value_type& get() const {
    return x;
  }

private:
  /**
   * The value.
   */
  value_type x;
};
}
