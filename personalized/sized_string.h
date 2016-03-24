// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)
#pragma once
#include <utility>
#include <vector>
#include <string>
#include "arena.h"
#include "base/string_util.h"
#include "base/hash.h"


namespace personalized{

#define DEFINE_STRING(string_name) \
const char* string_name = #string_name; \
const size_t string_name##_len = strlen(string_name); \

#define DECLARE_char_point(string_name) \
extern const char* string_name; \
extern const size_t string_name##_len; \

#define ADD_BUFFER(arena, mem_p, sized_string, buffer, len, features, weight) \
  mem_p = arena->Allocate(len); \
  sized_string.set(mem_p, len); \
  memcpy(mem_p, buffer, len); \
  features.push_back(std::make_pair(sized_string, weight)); \

#define COPY_STRING(arena, p, sized_string, string) \
  p = arena->Allocate(string.size()); \
  memcpy(p, string.c_str(), string.size()); \
  sized_string.set(p, string.size()); \

#define COPY_BUFFER(arena, p, sized_string, start, len) \
  p = arena->Allocate(len); \
  memcpy(p, start, len); \
  sized_string.set(p, len); \

#define ADD_CONST_STRING(sized_string, p, len, features, weight) \
sized_string.set(p, len); \
features.push_back(std::make_pair(sized_string, weight)); \



//struct sized string

struct SizedString
{
  union
  {
    const uint8 *pbData;
    const char  *pcData;
  };
  size_t cbData;

  // method
  SizedString();
  SizedString(const std::string& str, size_t start = 0);
  SizedString(char* p, size_t l);
  void set(const char* p, size_t l) {
    pcData = p;
    cbData = l;
  }

  bool operator== (const SizedString& other) const
  {
    if (cbData != other.cbData) {
      return false;
    }

    const uint32* p1 = (const uint32 *)pcData;
    const uint32* p2 = (const uint32 *)other.pcData;
    const uint32* end1 = (const uint32 *)pcData + (cbData/4);

    while (p1 != end1) {
      if (*p1 != *p2) {
        return false;
      }
      p1++;
      p2++;
    }

    const uint8* data1 = (const uint8*)p1;
    const uint8* data2 = (const uint8*)p2;
    for (int i = 0; i < (cbData & 3); i++) {
      if (*data1 != *data2) {
        return false;
      }
      data1++;
      data2++;
    }
    return true;
  }

  size_t operator()(const SizedString& string) const
  {
    size_t seed = string.cbData;
    return base::MurmurHash32A(string.pcData, string.cbData, seed);
  }

};

// sized string builder
struct SizedStringBuilder {
  uint32 size;
  SizedString builder[4];
  float weight;

  bool operator== (const SizedStringBuilder& other) const
  {
    if (size != other.size) {
      return false;
    }
    for (int i = 0; i < size; i++) {
      if (!(builder[i] == other.builder[i])) {
        return false;
      }
    }
    return true;
  }

  size_t operator()(const SizedStringBuilder& string) const
  {
    size_t seed = 49993;
    for (int i = 0; i < string.size; i++) {
      seed = base::MurmurHash32A(string.builder[i].pcData, string.builder[i].cbData, seed);
    }
    return seed;
  }

  std::string str() const;
  bool load(const std::string& line, Arena* arena);

};

// class string builder

class StringBuilder {
public:
  StringBuilder(Arena* arena);
  inline void extend(size_t len);
  inline SizedString append(const char* p1, size_t len1, const char* p2, size_t
  len2);
  inline SizedString append(const char* p1, size_t len1, const char* p2, size_t len2, const
            char* p3, size_t len3);
  inline SizedString append(const char* p1, size_t len1, const char* p2, size_t len2, const
              char* p3, size_t len3, const char* p4, size_t len4);
  inline void clear();


private:
  uint32 slice_idx_;
  uint32 slice_pointer_;
  Arena* arena_;
  std::vector<std::pair<char*, size_t> > buffer_;
};



} // namespace personalized

