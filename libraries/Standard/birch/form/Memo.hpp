/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Equips a form with a memoization facility.
 */
template<argument Middle>
struct Memo {
  using value_type = std::decay_t<decltype(eval(std::declval<Middle>()))>;

  /**
   * Memoized result.
   */
  std::optional<value_type> x;

  /**
   * Memoized form.
   */
  mutable Middle m;
 
  auto operator->() {
    return this;
  }
 
  auto operator->() const {
    return this;
  }
 
  operator auto() const;
  auto operator*() const;

  /**
   * Set the memoized value. This overrides const-ness to set the
   * value.
   */
  void set(const value_type& x) const {
    this->x = x;
  }

  /**
   * Set the memoized value. This overrides const-ness to set the
   * value.
   */
  void set(value_type&& x) const {
    this->x = std::move(x);
  }

  MEMBIRCH_STRUCT(Memo)
  MEMBIRCH_STRUCT_MEMBERS(m)
};

template<argument Middle>
struct is_form<Memo<Middle>> {
  static constexpr bool value = true;
};

template<argument Middle>
struct tag_s<Memo<Middle>> {
  using type = Memo<tag_t<Middle>>;
};

template<argument Middle>
struct peg_s<Memo<Middle>> {
  using type = Memo<peg_t<Middle>>;
};

template<argument Middle>
int rows(const Memo<Middle>& o) {
  return rows(eval(o));
}

template<argument Middle>
int columns(const Memo<Middle>& o) {
  return columns(eval(o));
}

template<argument Middle>
void reset(Memo<Middle>& o) {
  o.x.reset();
  reset(o.m);
}

template<argument Middle>
void relink(Memo<Middle>& o, const RelinkVisitor& visitor) {
  relink(o.m, visitor);
}

template<argument Middle>
void constant(const Memo<Middle>& o) {
  constant(o.m);
}

template<argument Middle>
bool is_constant(const Memo<Middle>& o) {
  return is_constant(o.m);
}

template<argument Middle>
void move(const Memo<Middle>& o, const MoveVisitor& visitor) {
  move(o.m, visitor);
  o.set(eval(o.m));  // update memoized result
}

template<argument Middle>
void args(Memo<Middle>& o, const ArgsVisitor& visitor) {
  args(o.m, visitor);
}

template<argument Middle>
void deep_grad(Memo<Middle>& o, const GradVisitor& visitor) {
  deep_grad(o.m, visitor);
}

template<argument Middle>
auto value(const Memo<Middle>& o) {
  if (!o.x) {
    o.set(value(o.m));
  }
  return slice(o.x);
}

template<argument Middle>
auto eval(const Memo<Middle>& o) {
  if (!o.x) {
    o.set(eval(o.m));
  }
  return slice(o.x);
}

template<argument Middle>
Memo<Middle>::operator auto() const {
  if (!x) {
    set(value(m));
  }
  return slice(*x);
}

template<argument Middle>
auto Memo<Middle>::operator*() const {
  if (!x) {
    set(value(m));
  }
  return wait(slice(*x));
}

}
