/**
 * @file
 */
#pragma once

#include "birch/basic.hpp"
#include "birch/utility.hpp"

namespace birch {

template<class... Args>
struct Base {
  std::tuple<Args...> tup;
 
  MEMBIRCH_STRUCT(Base)
  MEMBIRCH_STRUCT_MEMBERS(tup)

  template<class... Args1>
  Base(std::in_place_t, Args1&&... args) :
      tup(std::forward<Args1>(args)...) {
    //
  }

  template<class... Args1>
  Base(const Base<Args1...>& o) :
      tup(std::apply([]<class... Args2>(Args2&&... args) {
            return std::tuple<Args...>(std::forward<Args2>(args)...);
          }, o.tup)) {
    //
  }

  template<class... Args1>
  Base(Base<Args1...>&& o) :
      tup(std::apply([]<class... Args2>(Args2&&... args) {
            return std::tuple<Args...>(std::forward<Args2>(args)...);
          }, std::move(o.tup))) {
    //
  }

  Base(const Base&) = default;
  Base(Base&&) = default;

  auto& l() const {
    return std::get<0>(tup);
  }

  auto& m() const {
    return std::get<0>(tup);
  }

  auto& r() const {
    return std::get<1>(tup);
  }

  void reset() {
    std::apply([](auto&&... args) {
          (birch::reset(args), ...);
        }, tup);
  }
 
  void relink(const RelinkVisitor& visitor) {
    std::apply([&visitor](auto&&... args) {
          (birch::relink(args, visitor), ...);
        }, tup);
  }
 
  void constant() const {
    std::apply([](auto&&... args) {
          (birch::constant(args), ...);
        }, tup);
  }
 
  bool isConstant() const {
    return std::apply([](auto&&... args) {
        return (birch::is_constant(args) && ...);
      }, tup);
  }
 
  void move(const MoveVisitor& visitor) const {
    std::apply([&visitor](auto&&... args) {
        (birch::move(args, visitor), ...);
      }, tup);
  }

  void args(const ArgsVisitor& visitor) const {
    std::apply([&visitor](auto&&... args) {
        (birch::args(args, visitor), ...);
      }, tup);
  }

  void deepGrad(const GradVisitor& visitor) const {
    std::apply([&visitor](auto&&... args) {
        (birch::deep_grad(args, visitor), ...);
      }, tup);
  }
};

}
