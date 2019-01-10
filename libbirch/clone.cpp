/**
 * @file
 */
#include "libbirch/clone.hpp"

static bi::SharedPtr<bi::Memo> rootMemo = bi::Memo::create();
bi::SharedPtr<bi::Memo> bi::currentContext = rootMemo.get();
bool bi::cloneUnderway = false;
