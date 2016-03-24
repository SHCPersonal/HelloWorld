// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include <math.h>
#include <ctime>
#include <fstream>
#include <utility>
#include "arena.h"
#include "sized_string.h"
#include "ctr_predict_feature_builder.h"
#include "base/mutex.h"
#include "base/callback.h"
#include "base/hash.h"
#include "recommendation/online/engine/common/user_profile_util.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/datatypes_types.h"
#include "util/file_monitor/monitor_handler.h"
#include "shared/serving/serving_context/backend_serving_context.h"
#include "third_party/google/dense_hash_map"
#include "third_party/google/sparse_hash_map"
#include "third_party/gold_hash_map/gold_hash_map.hpp"


namespace shared {
  class ItemFeature;
}

namespace personalized{

typedef std::map<std::string, float> RawFeatureData;
typedef std::map<std::string, float>::const_iterator ConstIterRawFeatureData;
typedef std::map<std::string, float>::iterator IterRawFeatureData;

typedef gold_hash_map<SizedStringBuilder, float, personalized::SizedStringBuilder> ModelData;
typedef gold_hash_map<SizedStringBuilder, float, personalized::SizedStringBuilder>::const_iterator IterModelData;
typedef boost::unordered_set<SizedString, personalized::SizedString> FeatureData;
typedef boost::unordered_set<SizedString, personalized::SizedString>::const_iterator IterFeatureData;

struct CTR_feature;
typedef google::dense_hash_map<std::string, CTR_feature > CtrData;

class Classifier {
public:
  Classifier() {};
  virtual bool Init(const std::map<std::string, std::string>& parameter) = 0;
  virtual bool GetCTRFeature(const std::string& vid,
                     CTR_feature& ctr_feature) = 0;
  virtual float Predict(const std::vector<SizedStringBuilder >& feature_list) = 0;
  virtual void Debug(const std::vector<SizedStringBuilder >& feature_list,
             std::vector<std::string>& feature_res) = 0;
  virtual bool IsContainUser(SizedString& feature) = 0;
  virtual bool IsContainItem(SizedString& feature) = 0;
  //bool GetUserFeatureData(SizedString& feature_name);
  //bool GetItemFeatureData(SizedString& feature_name);
  virtual ~Classifier() {};
  virtual std::string GetFeatureBuilderType() = 0;
};


class LogisticRegression
  : public Classifier {
 public:
  LogisticRegression();
  bool Init(const std::map<std::string, std::string>& parameter);
  bool GetCTRFeature(const std::string& vid,
                     CTR_feature& ctr_feature);
  float Predict(const std::vector<SizedStringBuilder >& feature_list);
  void Debug(const std::vector<SizedStringBuilder >& feature_list,
             std::vector<std::string>& feature_res);
  bool IsContainUser(SizedString& feature);
  bool IsContainItem(SizedString& feature);
  //bool GetUserFeatureData(SizedString& feature_name);
  //bool GetItemFeatureData(SizedString& feature_name);
  virtual ~LogisticRegression() {}
  virtual std::string GetFeatureBuilderType(); 

 private:
  void RefreshModel();
  bool LoadModel(const std::string& model_file, ModelData* model_data, FeatureData*
                 user_feature, FeatureData* item_feature, Arena* arena);
  void RefreshCTR();
  bool LoadCTR(CtrData* ctr_data);

 private:
  bool init_bool_;
  base::RwMutex model_rwmutex_;
  base::RwMutex ctr_data_rwmutex_;
  base::Mutex  init_mutex_;
  scoped_ptr<CtrData> ctr_data_;
  scoped_ptr<FeatureData> user_feature_set_;
  scoped_ptr<FeatureData> item_feature_set_;
  scoped_ptr<ModelData> model_data_;
  scoped_ptr<Arena> model_memory_;
  util::FileMonitor* file_monitor_p_;

  std::string model_type_string_;
  std::string model_file_string_;
  std::string ctr_data_file_string_;

  std::string n_ctr_rate_string_;
  std::string y_ctr_rate_string_;

  std::string feature_builder_type_;

 private:
  DISALLOW_COPY_AND_ASSIGN(LogisticRegression);
};

} //namespace personalized

