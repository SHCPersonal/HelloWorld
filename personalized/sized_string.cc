// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "sized_string.h"
#include "base/hash.h"
#include "ctr_predict_feature_builder.h"
#include "iostream"

namespace personalized {



// struct sized_string

SizedString::SizedString(): pcData(NULL), cbData(0) {}

SizedString::SizedString(const std::string& str, size_t start) {
  pcData = str.c_str() + start;
  cbData = str.size() - start;
}

SizedString::SizedString(char* p, size_t l)
  :pcData(p),
  cbData(l){}

// sized string builder

std::string SizedStringBuilder::str() const {
  std::ostringstream res_str;
  for (int j = 0; j < size; j++) {
    std::string tmp_stl_string(builder[j].pcData, builder[j].cbData);
    res_str << tmp_stl_string;
    if (j != size - 1) {
      res_str << string_splitor_c;
    }
  }
  return res_str.str();
}

bool SizedStringBuilder::load(const std::string& line, Arena* arena) {
  char* p;
  SizedString tmp_str;
  std::vector<std::string> items;
  std::vector<std::string> feature_strings;
  SplitString(line, '\t', &items);
  if (items.size() < 2) {
    LOG(ERROR) << "CTR predict format error: " << line;
    return false;
  }
  SplitString(items[0], string_splitor_c, &feature_strings);
  if (feature_strings.size() == 0) {
    return false;
  }

  double value;
  if (!StringToDouble(items[1], &value)) {
    LOG(ERROR) << "CTR predict string parser error: " << items[1];
    return false;
  }

  size = feature_strings.size();
  /*if (feature_strings[0][0] <= '9') {
    if ((size - 1) != (feature_strings[0].size() / 2)) {
      LOG(ERROR) << "CTR predict model key error";
    }
  }*/
  for (int i = 0; i < size; i++) {
    COPY_STRING(arena, p, tmp_str, feature_strings[i])
    builder[i] = tmp_str;
  }
  weight = value;
  return true;
}

// string builder

StringBuilder::StringBuilder(Arena* arena)
  :slice_idx_(0),slice_pointer_(0),arena_(arena) {

  buffer_.reserve(8);
  size_t buffer_size = 1024 << 3;
  char* p = arena_->AllocateAligned(buffer_size);
  buffer_.push_back(std::make_pair(p, buffer_size));
}

void StringBuilder::extend(size_t len) {
  while (slice_pointer_ + len > buffer_[slice_idx_].second &&
         slice_idx_ < buffer_.size()) {
    slice_idx_++;
  }
  if (slice_idx_ == buffer_.size()) {
    size_t new_size, allocate_size;
    char* p;
    new_size = buffer_.back().second << 1;

    if (len <= new_size) {
      p = arena_->AllocateAligned(new_size);
      allocate_size = new_size;
    } else {
      p = arena_->Allocate(len);
      allocate_size = len;
    }

    buffer_.push_back(std::make_pair(p, allocate_size));
  }
}

SizedString StringBuilder::append(const char* p1, size_t len1, const char* p2, size_t len2) {
  size_t len = len1 + len2;
  if (slice_pointer_ + len > buffer_[slice_idx_].second) {
    extend(len);
  }
  memcpy(buffer_[slice_idx_].first + slice_pointer_, p1, len1);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1, p2, len2);
  SizedString tmp_str(buffer_[slice_idx_].first + slice_pointer_, len);
  slice_pointer_ += len;
  return tmp_str;
}

SizedString StringBuilder::append(const char* p1, size_t len1, const char* p2, size_t len2, const
            char* p3, size_t len3) {
  size_t len = len1 + len2 + len3;
  if (slice_pointer_ + len > buffer_[slice_idx_].second) {
    extend(len);
  }
  memcpy(buffer_[slice_idx_].first + slice_pointer_, p1, len1);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1, p2, len2);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1 + len2, p3, len3);
  SizedString tmp_str(buffer_[slice_idx_].first + slice_pointer_, len);
  slice_pointer_ += len;
  return tmp_str;
}

SizedString StringBuilder::append(const char* p1, size_t len1, const char* p2, size_t len2, const
            char* p3, size_t len3, const char* p4, size_t len4) {
  size_t len = len1 + len2 + len3 + len4;
  if (slice_pointer_ + len > buffer_[slice_idx_].second) {
    extend(len);
  }
  memcpy(buffer_[slice_idx_].first + slice_pointer_, p1, len1);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1, p2, len2);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1 + len2, p3, len3);
  memcpy(buffer_[slice_idx_].first + slice_pointer_ + len1 + len2 + len3, p4,
         len4);
  SizedString tmp_str(buffer_[slice_idx_].first + slice_pointer_, len);
  slice_pointer_ += len;
  return tmp_str;
}

void StringBuilder::clear() {
  slice_idx_ = 0;
  slice_pointer_ = 0;
}



}

