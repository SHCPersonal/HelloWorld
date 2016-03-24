// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)
#pragma once
#include "base/hash.h"

namespace personalized {

extern uint64 Hash(const char* data, size_t& n, uint32& seed);
extern uint64 Hash2(const void* data1, size_t& n1, const void* data2, size_t& n2, uint32& seed);
extern uint64 Hash3(const void* data1, size_t& n1, const void* data2, size_t& n2, const void* data3, size_t& n3, uint32& seed);

} //namespace personalized

