/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/TypeParameter.hpp"

template<class Target>
bi::Reference<Target>::Reference(Target* target) :
    target(target) {
  //
}

template<class Target>
bi::Reference<Target>::~Reference() {
  //
}

/*
 * Explicit instantiations.
 */
template class bi::Reference<bi::VarParameter>;
template class bi::Reference<bi::FuncParameter>;
template class bi::Reference<bi::TypeParameter>;
template class bi::Reference<bi::ProgParameter>;
