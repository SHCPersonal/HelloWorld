// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "hash_wrapper.h"

namespace personalized {

uint64 Hash2(const void* data1, size_t& n1, const void* data2, size_t& n2, uint32& seed){
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64 h = seed ^ ((n1 + n2) * m);

  const uint64* data = (const uint64 *)data1;
  const uint64* end = (const uint64 *)data1 + (n1/8);

  while (data != end) {
    uint64 k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const uint8* _data = (const uint8*)data;

  switch (n1 & 7) {
    case 7: h ^= static_cast<uint64>(_data[6]) << 48;
    case 6: h ^= static_cast<uint64>(_data[5]) << 40;
    case 5: h ^= static_cast<uint64>(_data[4]) << 32;
    case 4: h ^= static_cast<uint64>(_data[3]) << 24;
    case 3: h ^= static_cast<uint64>(_data[2]) << 16;
    case 2: h ^= static_cast<uint64>(_data[1]) << 8;
    case 1: h ^= static_cast<uint64>(_data[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  data = (const uint64 *)data2;
  end = (const uint64 *)data2 + (n2/8);
  while (data != end) {
    uint64 k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  _data = (const uint8*)data;

  switch (n2 & 7) {
    case 7: h ^= static_cast<uint64>(_data[6]) << 48;
    case 6: h ^= static_cast<uint64>(_data[5]) << 40;
    case 5: h ^= static_cast<uint64>(_data[4]) << 32;
    case 4: h ^= static_cast<uint64>(_data[3]) << 24;
    case 3: h ^= static_cast<uint64>(_data[2]) << 16;
    case 2: h ^= static_cast<uint64>(_data[1]) << 8;
    case 1: h ^= static_cast<uint64>(_data[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64 Hash3(const void* data1, size_t& n1, const void* data2,
                                       size_t& n2, const void* data3, size_t& n3, uint32& seed) {
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64 h = seed ^ ((n1 + n2 + n3) * m);

  const uint64* data = (const uint64 *)data1;
  const uint64* end = (const uint64 *)data1 + (n1/8);

  while (data != end) {
    uint64 k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const uint8* _data = (const uint8*)data;

  switch (n1 & 7) {
    case 7: h ^= static_cast<uint64>(_data[6]) << 48;
    case 6: h ^= static_cast<uint64>(_data[5]) << 40;
    case 5: h ^= static_cast<uint64>(_data[4]) << 32;
    case 4: h ^= static_cast<uint64>(_data[3]) << 24;
    case 3: h ^= static_cast<uint64>(_data[2]) << 16;
    case 2: h ^= static_cast<uint64>(_data[1]) << 8;
    case 1: h ^= static_cast<uint64>(_data[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  data = (const uint64 *)data2;
  end = (const uint64 *)data2 + (n2/8);
  while (data != end) {
    uint64 k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  _data = (const uint8*)data;

  switch (n2 & 7) {
    case 7: h ^= static_cast<uint64>(_data[6]) << 48;
    case 6: h ^= static_cast<uint64>(_data[5]) << 40;
    case 5: h ^= static_cast<uint64>(_data[4]) << 32;
    case 4: h ^= static_cast<uint64>(_data[3]) << 24;
    case 3: h ^= static_cast<uint64>(_data[2]) << 16;
    case 2: h ^= static_cast<uint64>(_data[1]) << 8;
    case 1: h ^= static_cast<uint64>(_data[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  data = (const uint64 *)data3;
  end = (const uint64 *)data3 + (n3/8);
  while (data != end) {
    uint64 k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  _data = (const uint8*)data;

  switch (n3 & 7) {
    case 7: h ^= static_cast<uint64>(_data[6]) << 48;
    case 6: h ^= static_cast<uint64>(_data[5]) << 40;
    case 5: h ^= static_cast<uint64>(_data[4]) << 32;
    case 4: h ^= static_cast<uint64>(_data[3]) << 24;
    case 3: h ^= static_cast<uint64>(_data[2]) << 16;
    case 2: h ^= static_cast<uint64>(_data[1]) << 8;
    case 1: h ^= static_cast<uint64>(_data[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}



uint64 Hash(const char* data, size_t& n, uint32& seed) {
  // Similar to murmur hash
  //const uint32 seed = 0x1c3567a7;
  const uint32 m = 0xc6a4a793;
  const uint32 r = 24;
  const char* limit = data + n;
  uint32 h = seed ^ (n * m);
  uint32 res[2] = {h, ~h};
  uint32 idx=0;

  uint32_t w;
  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    w = ((static_cast<uint32_t>(static_cast<unsigned char>(data[0])))
        | (static_cast<uint32_t>(static_cast<unsigned char>(data[1])) << 8)
        | (static_cast<uint32_t>(static_cast<unsigned char>(data[2])) << 16)
        | (static_cast<uint32_t>(static_cast<unsigned char>(data[3])) << 24));
    data += 4;
    res[idx] += w;
    res[idx] *= m;
    res[idx] ^= (res[idx] >> 16);
    idx = (idx + 1) & 1;
  }

  // Pick up remaining bytes
  switch (limit - data) {
    case 3:
      res[idx] += data[2] << 16;
      // fall through
    case 2:
      res[idx] += data[1] << 8;
      // fall through
    case 1:
      res[idx] += data[0];
      res[idx] *= m;
      res[idx] ^= (res[idx] >> r);
      break;
  }
  return (static_cast<uint64>(res[0]) << 32) | static_cast<uint64>(res[1]);
}


} // namespace personalized

