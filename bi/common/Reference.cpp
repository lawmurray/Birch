/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/ModelParameter.hpp"

template<class Target>
bi::Reference<Target>::Reference(const Target* target) :
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
template class bi::Reference<bi::ModelParameter>;
template class bi::Reference<bi::ProgParameter>;
