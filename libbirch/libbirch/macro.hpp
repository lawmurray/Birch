/**
 * @file
 */
#pragma once

/**
 * @internal
 *
 * @def LIBBIRCH_BASE
 *
 * Declare the base type for a class.
 */
#define LIBBIRCH_BASE(Base...) \
  using base_type_ = Base; \
  \
  auto shared_from_base_() { \
    return libbirch::Shared<base_type_>(static_cast<base_type_*>(this)); \
  }

/**
 * @internal
 *
 * @def LIBBIRCH_THIS
 *
 * Declare this type for a class.
 */
#define LIBBIRCH_THIS(Name...) \
  using this_type_ = Name; \
  \
  auto shared_from_this_() { \
    return libbirch::Shared<this_type_>(this); \
  }

/**
 * @internal
 *
 * @def LIBBIRCH_VIRTUAL
 *
 * Declare virtual functions for concrete classes.
 */
#define LIBBIRCH_VIRTUAL(Name, Base...) \
  virtual const char* getClassName_() const override { \
    return #Name; \
  } \
  \
  virtual Name* copy_() const override { \
    return new Name(*this); \
  }

/**
 * @def LIBBIRCH_CLASS
 *
 * Boilerplate macro for classes to support lazy deep copy. The first
 * argument is the name of the class; this should exclude any generic type
 * arguments. The second argument is the base class; this should include any
 * generic type arguments.
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
  LIBBIRCH_THIS(Name) \
  LIBBIRCH_BASE(Base) \
  LIBBIRCH_VIRTUAL(Name, Base)

/**
 * @def LIBBIRCH_ABSTRACT_CLASS
 *
 * Use in place of LIBBIRCH_CLASS when the containing class is abstract.
 */
#define LIBBIRCH_ABSTRACT_CLASS(Name, Base...) \
  LIBBIRCH_THIS(Name) \
  LIBBIRCH_BASE(Base)

/**
 * @def LIBBIRCH_CLASS_MEMBERS
 *
 * Boilerplate macro for classes to support lazy deep copy. The arguments
 * list all member variables of the class (but not those of a base class,
 * which should be listed in its own use of LIBBIRCH_CLASS_MEMBERS).
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
#define LIBBIRCH_CLASS_MEMBERS(...) \
  void accept_(libbirch::Marker& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Scanner& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Reacher& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Collector& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::BiconnectedCollector& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) override { \
    int l_, h_, m_, l1_, h1_, m1_; \
    std::tie(l_, h_, m_) = base_type_::accept_(visitor_, i_, j_); \
    std::tie(l1_, h1_, m1_) = visitor_.visit(i_, j_ + m_, __VA_ARGS__); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    return std::make_tuple(l_, h_, m_); \
  } \
  \
  std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) override { \
    int l_, h_, m_, n_, l1_, h1_, m1_, n1_; \
    std::tie(l_, h_, m_, n_) = base_type_::accept_(visitor_, j_, k_); \
    std::tie(l1_, h1_, m1_, n1_) = visitor_.visit(j_ + m_, k_ + n_, __VA_ARGS__); \
    l_ = std::min(l_, l1_); \
    h_ = std::max(h_, h1_); \
    m_ += m1_; \
    n_ += n1_; \
    return std::make_tuple(l_, h_, m_, n_); \
  } \
  \
  void accept_(libbirch::Copier& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::BiconnectedCopier& visitor_) override { \
    base_type_::accept_(visitor_); \
    visitor_.visit(__VA_ARGS__); \
  }

/**
 * @def LIBBIRCH_CLASS_NO_MEMBERS
 * 
 * Alternative to LIBBIRCH_CLASS_MEMBERS for classes with no members.
 */
#define LIBBIRCH_CLASS_NO_MEMBERS() \
  void accept_(libbirch::Marker& visitor_) override { \
    base_type_::accept_(visitor_); \
  } \
  \
  void accept_(libbirch::Scanner& visitor_) override { \
    base_type_::accept_(visitor_); \
  } \
  \
  void accept_(libbirch::Reacher& visitor_) override { \
    base_type_::accept_(visitor_); \
  } \
  \
  void accept_(libbirch::Collector& visitor_) override { \
    base_type_::accept_(visitor_); \
  } \
  \
  void accept_(libbirch::BiconnectedCollector& visitor_) override { \
    base_type_::accept_(visitor_); \
  } \
  \
  std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) override { \
    return base_type_::accept_(visitor_, i_, j_); \
  } \
  \
  std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) override { \
    return base_type_::accept_(visitor_, j_, k_); \
  } \
  \
  void accept_(libbirch::Copier& visitor_) override { \
    return base_type_::accept_(visitor_); \
  } \
  \
  void accept_(libbirch::BiconnectedCopier& visitor_) override { \
    return base_type_::accept_(visitor_); \
  }

/**
 * @def LIBBIRCH_STRUCT
 *
 * Boilerplate macro for structs to support lazy deep copy. The only argument
 * is the name of the struct; this should exclude any generic type
 * arguments.
 *
 * LIBBIRCH_STRUCT must be followed by LIBBIRCH_STRUCT_MEMBERS, and should be
 * in a public section.
 */
#define LIBBIRCH_STRUCT(Name)

/**
 * @def LIBBIRCH_STRUCT_MEMBERS
 *
 * Boilerplate macro for structs to support lazy deep copy. The arguments
 * list all member variables of the class.
 *
 * LIBBIRCH_STRUCT_MEMBERS must be preceded by LIBBIRCH_STRUCT, and should be
 * in a public section.
 */
#define LIBBIRCH_STRUCT_MEMBERS(...) \
  void accept_(libbirch::Marker& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Scanner& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Reacher& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Collector& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::BiconnectedCollector& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) { \
    return visitor_.visit(i_, j_, __VA_ARGS__); \
  } \
  \
  std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) { \
    return visitor_.visit(j_, k_, __VA_ARGS__); \
  } \
  \
  void accept_(libbirch::Copier& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  } \
  \
  void accept_(libbirch::BiconnectedCopier& visitor_) { \
    visitor_.visit(__VA_ARGS__); \
  }

/**
 * @def LIBBIRCH_STRUCT_NO_MEMBERS
 * 
 * Alternative to LIBBIRCH_STRUCT_MEMBERS for structs with no members.
 */
#define LIBBIRCH_STRUCT_NO_MEMBERS() \
  void accept_(libbirch::Marker& visitor_) {} \
  void accept_(libbirch::Scanner& visitor_) {} \
  void accept_(libbirch::Reacher& visitor_) {} \
  void accept_(libbirch::Collector& visitor_) {} \
  void accept_(libbirch::BiconnectedCollector& visitor_) {} \
  \
  std::tuple<int,int,int> accept_(libbirch::Spanner& visitor_, const int i_, const int j_) { \
    return std::make_tuple(i_, i_, 0); \
  } \
  \
  std::tuple<int,int,int,int> accept_(libbirch::Bridger& visitor_, const int j_, const int k_) { \
    return std::make_tuple(libbirch::Bridger::MAX, 0, 0, 0); \
  } \
  \
  void accept_(libbirch::Copier& visitor_) {} \
  void accept_(libbirch::BiconnectedCopier& visitor_) {}

#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Reacher.hpp"
#include "libbirch/Collector.hpp"
#include "libbirch/BiconnectedCollector.hpp"
#include "libbirch/Spanner.hpp"
#include "libbirch/Bridger.hpp"
#include "libbirch/Copier.hpp"
#include "libbirch/BiconnectedCopier.hpp"
