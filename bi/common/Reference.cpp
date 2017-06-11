/**
 * @file
 */
#include "bi/common/Reference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/BinaryParameter.hpp"
#include "bi/expression/UnaryParameter.hpp"
#include "bi/type/TypeParameter.hpp"
#include "bi/expression/ProgParameter.hpp"

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
template class bi::Reference<bi::BinaryParameter>;
template class bi::Reference<bi::UnaryParameter>;
template class bi::Reference<bi::TypeParameter>;
template class bi::Reference<bi::ProgParameter>;
