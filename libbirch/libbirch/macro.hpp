/**
 * @file
 */
#pragma once

namespace libbirch {
template<class T, class ...Args, std::enable_if_t<!std::is_abstract<T>::value,int> = 0>
auto make_object(Args&&... args) {
  return new T(std::forward<Args>(args)...);
} 
template<class T, class ...Args, std::enable_if_t<std::is_abstract<T>::value,int> = 0>
auto make_object(Args&&... args) {
  return nullptr;
}

using no_base = void;
static int no_members = 0;
}

/**
 * @def LIBBIRCH_NO_BASE
 * 
 * Used as argument to LIBBIRCH_STRUCT when there is no base struct, e.g.
 * `LIBBIRCH_STRUCT(Name, LIBBIRCH_NO_BASE)`.
 */
#define LIBBIRCH_NO_BASE libbirch::no_base

/**
 * @def LIBBIRCH_NO_MEMBERS
 * 
 * Used as argument to LIBBIRCH_CLASS_MEMBERS or LIBBIRCH_STRUCT_MEMBERS when
 * there are no member variables, e.g.
 * `LIBBIRCH_CLASS_MEMBERS(LIBBIRCH_NO_MEMBERS)`.
 */
#define LIBBIRCH_NO_MEMBERS libbirch::no_members

/**
 * @def LIBBIRCH_CLASS
 *
 * Boilerplate macro for polymorphic classes to support lazy deep copy. The
 * first argument is the name of the class; this should exclude any generic
 * type arguments. The second argument is the base class; this should include
 * any generic type arguments. There should always be a base class; if nothing
 * else, use Any.
 *
 * LIBBIRCH_CLASS must be followed by LIBBIRCH_CLASS_MEMBERS, and should be in
 * a public section, e.g.:
 *
 *     class A : public B {
 *     private:
 *       int x, y, z;
 *
 *     public:
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_CLASS_MEMBERS(x, y, z)
 *     };
 *
 * The use of a variadic macro here supports base classes that contain
 * commas without special treatment, e.g.
 *
 *     LIBBIRCH_CLASS(A, B<T,U>)
 */
#define LIBBIRCH_CLASS(Name, Base...) \
  using this_type_ = Name; \
  using base_type_ = Base; \
  \
  friend class Marker; \
  friend class Scanner; \
  friend class Reacher; \
  friend class Collector; \
  friend class BiconnectedCollector; \
  friend class Spanner; \
  friend class Bridger; \
  friend class Copier; \
  friend class Memo; \
  friend class BiconnectedCopier; \
  friend class BiconnectedMemo; \
  friend class Destroyer; \
  \
  virtual const char* getClassName_() const override { \
    return #Name; \
  } \
  \
  virtual Name* copy_() const override { \
    return libbirch::make_object<Name>(*this); \
  }

/**
 * @def LIBBIRCH_CLASS_MEMBERS
 *
 * Boilerplate macro for polymorphic classes to support lazy deep copy. The
 * arguments list all member variables of the class (but not those of a base
 * class, which should be listed in its own use of LIBBIRCH_CLASS_MEMBERS). If
 * there are no member variables, use `LIBBIRCH_CLASS_MEMBERS(LIBBIRCH_NO_MEMBERS)`.
 *
 * LIBBIRCH_CLASS_MEMBERS must be preceded by LIBBIRCH_CLASS, and should be in
 * a public section, e.g.:
 *
 *     class A : public B {
 *     private:
 *       int x, y, z;
 *
 *     public:
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_CLASS_MEMBERS(x, y, z)
 *     };
 */
#define LIBBIRCH_CLASS_MEMBERS(members...) \
  template<class V, class... Args, class T = base_type_, std::enable_if_t<std::is_void<T>::value,int> = 0> \
  auto accept_base_(V& visitor_, Args&&... args) { \
    return visitor_.visit(std::forward<Args>(args)...); \
  } \
  \
  template<class V, class... Args, class T = base_type_, std::enable_if_t<!std::is_void<T>::value,int> = 0> \
  auto accept_base_(V& visitor_, Args&&... args) { \
    return T::accept_(visitor_, std::forward<Args>(args)...); \
  } \
  \
  virtual void accept_(libbirch::Marker& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::Scanner& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::Reacher& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::Collector& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::BiconnectedCollector& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) override { \
    int l_, h_, m_, l1_, h1_, m1_; \
    std::tie(l_, h_, m_) = accept_base_(visitor_, i_, j_); \
    std::tie(l1_, h1_, m1_) = visitor_.visit(i_, j_ + m_, members); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    return std::make_tuple(l_, h_, m_); \
  } \
  \
  virtual std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) override { \
    int l_, h_, m_, n_, l1_, h1_, m1_, n1_; \
    std::tie(l_, h_, m_, n_) = accept_base_(visitor_, j_, k_); \
    std::tie(l1_, h1_, m1_, n1_) = visitor_.visit(j_ + m_, k_ + n_, members); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    n_ += n1_; \
    return std::make_tuple(l_, h_, m_, n_); \
  } \
  \
  virtual void accept_(libbirch::Copier& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::BiconnectedCopier& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  virtual void accept_(libbirch::Destroyer& visitor_) override { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  }

/**
 * @def LIBBIRCH_STRUCT
 *
 * Boilerplate macro for non-polymorphic structs to support lazy deep copy.
 * The first argument is the name of the struct; this should exclude any
 * generic type arguments. The second argument is the base struct; this should
 * include any generic type arguments. If there is no base struct, use
 * `LIBBIRCH_STRUCT(Name, LIBBIRCH_NO_BASE)`.
 *
 * LIBBIRCH_STRUCT must be followed by LIBBIRCH_STRUCT_MEMBERS, and should be
 * in a public section, e.g.:
 *
 *     struct A : public B {
 *     private:
 *       int x, y, z;
 *
 *     public:
 *       LIBBIRCH_STRUCT(A, B)
 *       LIBBIRCH_STRUCT_MEMBERS(x, y, z)
 *     };
 *
 * The use of a variadic macro here supports base structs that contain commas
 * without special treatment, e.g.
 *
 *     LIBBIRCH_STRUCT(A, B<T,U>)
 */
#define LIBBIRCH_STRUCT(Name, Base...) \
  using this_type_ = Name; \
  using base_type_ = Base; \
  \
  friend class Marker; \
  friend class Scanner; \
  friend class Reacher; \
  friend class Collector; \
  friend class BiconnectedCollector; \
  friend class Spanner; \
  friend class Bridger; \
  friend class Copier; \
  friend class Memo; \
  friend class BiconnectedCopier; \
  friend class BiconnectedMemo; \
  friend class Destroyer; \
  \
  const char* getClassName_() const { \
    return #Name; \
  }

/**
 * @def LIBBIRCH_STRUCT_MEMBERS
 *
 * Boilerplate macro for non-polymorphic structs to support lazy deep copy.
 * The arguments list all member variables of the struct (but not those of a
 * base struct, which should be listed in its own use of
 * LIBBIRCH_STRUCT_MEMBERS). If there are no member variables, use
 * `LIBBIRCH_STRUCT_MEMBERS(LIBBIRCH_NO_MEMBERS)`.
 *
 * LIBBIRCH_STRUCT_MEMBERS must be preceded by LIBBIRCH_STRUCT, and should be
 * in a public section, e.g.:
 *
 *     struct A : public B {
 *     private:
 *       int x, y, z;
 *
 *     public:
 *       LIBBIRCH_STRUCT(A, B)
 *       LIBBIRCH_STRUCT_MEMBERS(x, y, z)
 *     };
 */
#define LIBBIRCH_STRUCT_MEMBERS(members...) \
  template<class V, class... Args, class T = base_type_, std::enable_if_t<std::is_void<T>::value,int> = 0> \
  auto accept_base_(V& visitor_, Args&&... args) { \
    return visitor_.visit(std::forward<Args>(args)...); \
  } \
  \
  template<class V, class... Args, class T = base_type_, std::enable_if_t<!std::is_void<T>::value,int> = 0> \
  auto accept_base_(V& visitor_, Args&&... args) { \
    return T::accept_(visitor_, std::forward<Args>(args)...); \
  } \
  \
  void accept_(libbirch::Marker& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::Scanner& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::Reacher& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::Collector& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::BiconnectedCollector& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) { \
    int l_, h_, m_, l1_, h1_, m1_; \
    std::tie(l_, h_, m_) = accept_base_(visitor_, i_, j_); \
    std::tie(l1_, h1_, m1_) = visitor_.visit(i_, j_ + m_, members); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    return std::make_tuple(l_, h_, m_); \
  } \
  \
  std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) { \
    int l_, h_, m_, n_, l1_, h1_, m1_, n1_; \
    std::tie(l_, h_, m_, n_) = accept_base_(visitor_, j_, k_); \
    std::tie(l1_, h1_, m1_, n1_) = visitor_.visit(j_ + m_, k_ + n_, members); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    n_ += n1_; \
    return std::make_tuple(l_, h_, m_, n_); \
  } \
  \
  void accept_(libbirch::Copier& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::BiconnectedCopier& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  } \
  \
  void accept_(libbirch::Destroyer& visitor_) { \
    accept_base_(visitor_); \
    visitor_.visit(members); \
  }
