// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include "arena.h"
#include "sized_string.h"
#include "logistic_regression.h"
#include "shared/serving/proto/datatypes_types.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/media_doc_info_types.h"
#include "shared/serving/manager/repository.h"
#include "third_party/google/sparse_hash_map"
#include "third_party/gold_hash_map/gold_hash_map.hpp"


namespace personalized {

class Classifier;


DECLARE_char_point(rate_value);
//DECLARE_char_point(rate_log);
//DECLARE_char_point(click_value);
//DECLARE_char_point(click_log);
//DECLARE_char_point(exp_value);
//DECLARE_char_point(exp_log);
DECLARE_char_point(y_recvid_rate);
DECLARE_char_point(n_recvid_rate);
DECLARE_char_point(release_month_1);
DECLARE_char_point(release_month_2);
DECLARE_char_point(release_month_3);
DECLARE_char_point(release_month_6);
DECLARE_char_point(release_month_24);
DECLARE_char_point(release_month_old);
DECLARE_char_point(release_null);
DECLARE_char_point(create_day_1);
DECLARE_char_point(create_day_2);
DECLARE_char_point(create_month_1);
DECLARE_char_point(create_month_2);
DECLARE_char_point(create_month_3);
DECLARE_char_point(create_month_6);
DECLARE_char_point(create_month_old);
DECLARE_char_point(create_null);

extern const std::string same_tag_prefix;
extern const char string_splitor_c;


#define BUFFERLEN 32
#define BUFFER_SIZE 1024


enum CTRPredictFeatureType {
  RELEASE_TIME = shared::FeatureType::FEATURE_UNKNOWN + 1,
  CREATE_TIME = shared::FeatureType::FEATURE_UNKNOWN + 2,
  HISTORY = shared::FeatureType::FEATURE_UNKNOWN + 3,
  ItemCF = shared::FeatureType::FEATURE_UNKNOWN + 4,
  AGE = shared::FeatureType::FEATURE_UNKNOWN + 5,
  GENDER = shared::FeatureType::FEATURE_UNKNOWN + 6,
  EDU = shared::FeatureType::FEATURE_UNKNOWN + 7,
  ALBUM_FOLLOW = shared::FeatureType::FEATURE_UNKNOWN + 8,
  FEATURN_COUNT = shared::FeatureType::FEATURE_UNKNOWN + 9
  //PID = shared::FeatureType::FEATURE_UNKNOWN + 4
};


extern const char string_splitor_c;

enum FeatureSource {
  FromUser  = 1,
  FromItem  = 2,
  FromTime  = 3,
  FeatureSourceNum = 4
};
enum WeightType {
  Bool = 1,
  Float = 2,
  UNKNOWN = 3
};
enum TimeFeatureType {
  Time = 1,
  Date = 2,
  TimeFeatureTypeNum = 3
};

class FeatureTypeMapping {
public:
  FeatureTypeMapping() :
    mapping_table(NULL) {
    mapping_table = new char [FEATURN_COUNT];
    for (size_t i = 0 ; i < FEATURN_COUNT; ++i) {
      mapping_table[i] = 0;
    }
    mapping_table[shared::FeatureType::AREA]='a';
    mapping_table[shared::FeatureType::CATEGORY]='c';
    mapping_table[shared::FeatureType::SUBCATEGORY]='s';
    mapping_table[shared::FeatureType::PEOPLE]='p';
    mapping_table[shared::FeatureType::TAG]='t';
    mapping_table[shared::FeatureType::LABEL]='l';
    mapping_table[shared::FeatureType::PID]='P';
    mapping_table[shared::FeatureType::CTR]='C';
    mapping_table[shared::FeatureType::VID]='v';
    mapping_table[shared::FeatureType::LEWORD]='L';

    mapping_table[personalized::RELEASE_TIME]='r';
    mapping_table[personalized::CREATE_TIME]='t';
    mapping_table[personalized::HISTORY]='h';
    mapping_table[personalized::ItemCF]='i';
    mapping_table[personalized::AGE]='A';
    mapping_table[personalized::GENDER]='g';
    mapping_table[personalized::EDU]='e';
    mapping_table[personalized::ALBUM_FOLLOW]='f';
  }
  virtual ~FeatureTypeMapping() {
    delete mapping_table;
  }

  char* mapping_table;
};

extern FeatureTypeMapping g_feature_mapping;

class FeatureBuildRule {
public:
  FeatureBuildRule() {};
  FeatureBuildRule(
          int _combine_num,
          int _feature1_type,
          WeightType    _feature1_weight_type,
          FeatureSource _feature1_source,
          int           _feature2_type = 0,
          WeightType    _feature2_weight_type = UNKNOWN,
          FeatureSource _feature2_source = FeatureSourceNum,
          int           _feature3_type = 0,
          WeightType    _feature3_weight_type = UNKNOWN,
          FeatureSource _feature3_source = FeatureSourceNum)
    : combine_num(_combine_num),
      feature1_type(_feature1_type),
      feature1_weight_type(_feature1_weight_type),
      feature1_source(_feature1_source),
      feature2_type(_feature2_type),
      feature2_weight_type(_feature2_weight_type),
      feature2_source(_feature2_source),
      feature3_type(_feature3_type),
      feature3_weight_type(_feature3_weight_type),
      feature3_source(_feature3_source) {
    switch(_combine_num) {
      case 1: {
        prefix_n = prefix_n << 1 | (feature1_weight_type & 0);
        prefix_str.push_back('0'+feature1_weight_type);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back('a'+feature1_type);
        /*prefix_str = StringPrintf("%d-%d-%d", feature1_weight_type,
                                              feature1_source,
                                              feature1_type);*/
        break;
      }
      case 2: {
        prefix_n  = ((feature1_source & 3) << 6) |  (feature1_type & 63)
                    | ((feature2_source & 3) << 14) |  ((feature2_type & 63) << 8);
        prefix_str.push_back('0'+feature1_weight_type);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back('a'+feature1_type);
        prefix_str.push_back('0'+feature2_source);
        prefix_str.push_back('a'+feature2_type);
        /*prefix_str = StringPrintf("%d-%d-%d-%d-%d", feature1_weight_type,
                                                    feature1_source,
                                                    feature1_type,
                                                    feature2_source,
                                                    feature2_type);*/
        break;
      }
      case 3: {
        prefix_n  = ((feature1_source & 3) << 6) &  (feature1_type & 63)
                    & ((feature2_source & 3) << 14) &  ((feature2_type & 63) << 8)
                    & ((feature3_source & 3) << 22) &  ((feature3_type & 63) << 16);
        prefix_str.push_back('0'+feature1_weight_type);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back('a'+feature1_type);
        prefix_str.push_back('0'+feature2_source);
        prefix_str.push_back('a'+feature2_type);
        prefix_str.push_back('0'+feature3_source);
        prefix_str.push_back('a'+feature3_type);
        /*prefix_str =  StringPrintf("%d-%d-%d-%d-%d-%d-%d", feature1_weight_type,
                                                        feature1_source,
                                                        feature1_type,
                                                        feature2_source,
                                                        feature2_type,
                                                        feature3_source,
                                                        feature3_type);*/
        break;
      }
    }
  }

  int combine_num;
  int feature1_type;
  WeightType feature1_weight_type;
  FeatureSource feature1_source;

  int feature2_type;
  WeightType feature2_weight_type;
  FeatureSource feature2_source;

  int feature3_type;
  WeightType feature3_weight_type;
  FeatureSource feature3_source;

  uint32 prefix_n;
  std::string prefix_str;
};

class PrefixMappingFeatureBuildRule
  : public FeatureBuildRule
{
public:
  PrefixMappingFeatureBuildRule(
          int _combine_num,
          int _feature1_type,
          WeightType    _feature1_weight_type,
          FeatureSource _feature1_source,
          int           _feature2_type = 0,
          WeightType    _feature2_weight_type = UNKNOWN,
          FeatureSource _feature2_source = FeatureSourceNum,
          int           _feature3_type = 0,
          WeightType    _feature3_weight_type = UNKNOWN,
          FeatureSource _feature3_source = FeatureSourceNum)
    {
    combine_num = _combine_num;
    feature1_type = _feature1_type;
    feature1_weight_type = _feature1_weight_type;
    feature1_source = _feature1_source;
    feature2_type = _feature2_type;
    feature2_weight_type = _feature2_weight_type;
    feature2_source = _feature2_source;
    feature3_type = _feature3_type;
    feature3_weight_type = _feature3_weight_type;
    feature3_source = _feature3_source;
    switch(_combine_num) {
      case 1: {
        prefix_n = prefix_n << 1 | (feature1_weight_type & 0);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature1_type]);
        CHECK(g_feature_mapping.mapping_table[feature1_type] != 0) << feature1_type;
        break;
      }
      case 2: {
        prefix_n  = ((feature1_source & 3) << 6) |  (feature1_type & 63)
                    | ((feature2_source & 3) << 14) |  ((feature2_type & 63) << 8);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature1_type]);
        CHECK(g_feature_mapping.mapping_table[feature1_type] != 0) << feature1_type;
        prefix_str.push_back('0'+feature2_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature2_type]);
        CHECK(g_feature_mapping.mapping_table[feature2_type] != 0) << feature2_type;
        break;
      }
      case 3: {
        prefix_n  = ((feature1_source & 3) << 6) &  (feature1_type & 63)
                    & ((feature2_source & 3) << 14) &  ((feature2_type & 63) << 8)
                    & ((feature3_source & 3) << 22) &  ((feature3_type & 63) << 16);
        prefix_str.push_back('0'+feature1_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature1_type]);
        CHECK(g_feature_mapping.mapping_table[feature1_type] != 0) << feature1_type;
        prefix_str.push_back('0'+feature2_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature2_type]);
        CHECK(g_feature_mapping.mapping_table[feature2_type] != 0) << feature2_type;
        prefix_str.push_back('0'+feature3_source);
        prefix_str.push_back(g_feature_mapping.mapping_table[feature3_type]);
        CHECK(g_feature_mapping.mapping_table[feature3_type] != 0) << feature3_type;
        break;
      }
    }
  }
};


struct CTR_feature {
  double rate;
  double click;
  double exp;

  double rate_value;
  double rate_log;
  uint32 rate_buckt;

  double click_value;
  double click_log;
  uint32 click_buckt;

  double exp_value;
  double exp_log;
  uint32 exp_buckt;

  bool Load(const std::string& line, std::string& vid) {
    std::vector<std::string> ctr_item;
    if (!line.empty()) {
      SplitString(line, ' ', &ctr_item);
      if (ctr_item.size() != 13) {
          LOG(ERROR) << "CTR preidct bad ctr line";
          LOG(ERROR) << line;
          return false;
      }

      vid = ctr_item[0];

      if (!StringToDouble(ctr_item[1], &rate)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[1];
        return false;
      }
      if (!StringToDouble(ctr_item[2], &click)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[2];
        return false;
      }
      if (!StringToDouble(ctr_item[3], &exp)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[3];
        return false;
      }
      if (!StringToDouble(ctr_item[4], &rate_value)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[4];
        return false;
      }
      if (!StringToDouble(ctr_item[5], &rate_log)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[5];
        return false;
      }
      if (!StringToUint(ctr_item[6], &rate_buckt)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[6];
        return false;
      }
      if (!StringToDouble(ctr_item[7], &click_value)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[7];
        return false;
      }
      if (!StringToDouble(ctr_item[8], &click_log)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[8];
        return false;
      }
      if (!StringToUint(ctr_item[9], &click_buckt)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[9];
        return false;
      }
      if (!StringToDouble(ctr_item[10], &exp_value)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[10];
        return false;
      }
      if (!StringToDouble(ctr_item[11], &exp_log)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[11];
        return false;
      }
      if (!StringToUint(ctr_item[12], &exp_buckt)) {
        LOG(ERROR) << "CTR preidct bad ctr value! " << ctr_item[12];
        return false;
      }
    }
    else {
      rate = -1.0;
      return false;
    }
    return true;
  }
};


class CTRPredictFeatureBuilder {
public:
  CTRPredictFeatureBuilder(bool for_training, Arena* arena);
  virtual ~CTRPredictFeatureBuilder();

  // feature builder
  void FeatureFromRawFeatureBuild(
      std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
      std::vector<SizedStringBuilder >& res_feature_vec);

  virtual void CustomizedFeatureBuild(std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
      std::vector<SizedStringBuilder >& res_feature_vec) = 0;

  // util function for feature builder
  void SameTagFeature(std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
      std::vector<SizedStringBuilder >& res_feature_vec);

  float GetSimAndSet(
      const std::string& prefix,
      std::vector<std::pair<SizedString, float> >& features1,
      std::vector<std::pair<SizedString, float> >& features2,
      std::vector<SizedStringBuilder >& res_feature_vec);

  inline void SingleFeatureGenerate(
       SizedString& prefix,
       SizedString& feature_name,
       float weight,
       std::vector<SizedStringBuilder >& res_feature_vec);
  inline void TwoFeatureGenerate(
       SizedString& prefix,
       SizedString& feature_name1,
       SizedString& feature_name2,
       float weight,
       std::vector<SizedStringBuilder >& res_feature_vec);
  inline void ThreeFeatureGenerate(
       SizedString& prefix,
       SizedString& feature_name1,
       SizedString& feature_name2,
       SizedString& feature_name3,
       float weight,
       std::vector<SizedStringBuilder >& res_feature_vec);

  std::string debug(std::vector<SizedStringBuilder >& res_feature_vec) {
    std::ostringstream res_str;

    for (int i = 0; i < res_feature_vec.size(); i++) {
      res_str << res_feature_vec[i].str();
      res_str << ":" << res_feature_vec[i].weight;
      if (i != res_feature_vec.size() -1) {
        res_str << "\t";
      }
    }
    return  res_str.str();
  }

  size_t GetStartOfString(const std::string& str);

  // load raw feature
  virtual void LoadUserProfileFromThrift(const profile::UserProfile* user_profile,
    Classifier* user_feature_set,
    //FeatureData* user_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& user_features_);
  virtual void LoadItemFeatureFromThrift(const serving::MediaDocInfo* media_doc_info,
    const CTR_feature* ctr_feature_p,
    Classifier* item_feature_set,
    //FeatureData* item_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
    const uint64& now);
  virtual void LoadItemFeatureFromThriftWithPid(const serving::MediaDocInfo* media_doc_info,
    const serving::MediaDocInfo* media_doc_info_pid,
    const CTR_feature* ctr_feature_p,
    Classifier* item_feature_set,
    //FeatureData* item_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
    const uint64& now) {};
  virtual void LoadTimeFeature(const std::string& time_str,
    std::vector<std::vector<std::pair<SizedString, float> > >& time_features_);
  virtual void LoadTimeFeature(const time_t& t,
    std::vector<std::vector<std::pair<SizedString, float> > >& time_features_);

  //use the new result media
  virtual void LoadItemFeatureFromResultMedia(const std::string& id,
                                              const serving::ResultMedia* media_doc_info,
                                              const CTR_feature* ctr_feature_p,
                                              Classifier* item_feature_set,
                                              //FeatureData* item_feature_set,
                                              std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
                                              const uint64& now) {}
  virtual std::string BuildFeatureOffline(
    const std::string& lc,
    const std::string& user_profile_string,
    const std::string& vid,
    const std::string& media_doc_info_string,
    const std::string& media_doc_info_pid_string,
    const std::string& vid_ctr,
    const std::string& time_t_string,
    std::vector<std::string>& history
    ) {return std::string();};

 protected:
  std::vector<FeatureBuildRule*> rules_;
  Arena* arena_;
  bool for_training_;
};


} // namespace personalized

