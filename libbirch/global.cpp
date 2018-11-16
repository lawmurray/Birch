/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/Memo.hpp"

bi::Memo* bi::cloneMemo = nullptr;
// ^ reserving a non-zero size seems necessary to avoid segfault here
