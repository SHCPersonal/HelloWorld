// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "general_recommender_feature_builder.h"
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "util/global_init/global_init.h"
#include "util/encode/base64encoder.h"


using namespace std;

namespace personalized {

GeneralRecommenderFeatureBuilder::GeneralRecommenderFeatureBuilder(bool for_training, Arena* arena)
  :CTRPredictFeatureBuilder(for_training_, arena){
  for_training_ = for_training;
  arena_ = arena;

  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::LEWORD, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::CATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::PEOPLE, Bool, FromItem));
  // pre compute in model
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));

  // history
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::HISTORY, Bool, FromUser,
                                       shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::HISTORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));*/

  // ctr feature
  /*rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::CTR, Float, FromItem,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));*/
  // intersection tag
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  // intersection subcategory
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));*/
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                     shared::FeatureType::VID, Bool, FromItem));*/

  // time user
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                       Time, Bool, FromTime));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       Time, Bool, FromTime));
  // time item
  rules_.push_back(new FeatureBuildRule(2, Time, Bool, FromTime,
                                       shared::FeatureType::CATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, Time, Bool, FromTime,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, Time, Bool, FromTime,
                                       shared::FeatureType::TAG, Bool, FromItem));*/
  // 3 intersect
  /*rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       Time, Bool, FromTime,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::CATEGORY, Bool, FromUser,
                                       Time, Bool, FromTime,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       Time, Bool, FromTime,
                                       shared::FeatureType::CATEGORY, Bool, FromItem));*/
  // release time and create time
  rules_.push_back(new FeatureBuildRule(1, personalized::RELEASE_TIME, Bool, FromItem));

  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                       personalized::RELEASE_TIME, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       personalized::RELEASE_TIME, Bool, FromItem));
}

void GeneralRecommenderFeatureBuilder::CustomizedFeatureBuild(
  std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
  std::vector<SizedStringBuilder >& res_feature_vec) {

  SameTagFeature(features, res_feature_vec);
}


std::string GeneralRecommenderFeatureBuilder::BuildFeatureOffline(
  const std::string& lc,
  const std::string& user_profile_string,
  const std::string& vid,
  const std::string& media_doc_info_string,
  const std::string& media_doc_info_pid_string,
  const std::string& vid_ctr,
  const std::string& time_t_string,
  std::vector<std::string>& history) {

  std::vector<std::vector<std::pair<SizedString, float> > > user_feature((int)personalized::FEATURN_COUNT);
  std::vector<std::vector<std::pair<SizedString, float> > > item_feature((int)personalized::FEATURN_COUNT);
  std::vector<std::vector<std::pair<SizedString, float> > > time_feature(TimeFeatureTypeNum);
  std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* > features(FeatureSourceNum);
  features[FromUser]=&user_feature;
  features[FromItem]=&item_feature;
  features[FromTime]=&time_feature;

  time_t now;
  CHECK(StringToInt64(time_t_string, &now)) << time_t_string;

  //deserialization user profile
  string user_profile_string_base64 = user_profile_string;

  util::Base64Encoder encoder;
  string info_line("");
  if (!encoder.Decode(user_profile_string_base64, &info_line)){
    CHECK(false);
  }
  profile::UserProfile user_profile;
  if (!base::FromStringToThrift(info_line, &user_profile)) {
    CHECK(false);
  }

  //deserialization media doc info
  string media_doc_info_string_base64 = media_doc_info_string;
  string media_info_line("");
  if (!encoder.Decode(media_doc_info_string_base64, &media_info_line)){
    CHECK(false);
  }
  serving::MediaDocInfo media_doc_info;
  if (!base::FromStringToThrift(media_info_line, &media_doc_info)) {
    CHECK(false);
  }

  //load user profile
  this->LoadUserProfileFromThrift(&user_profile, NULL, user_feature);
  // get user profile
  std::vector<std::string> user_history;
  for (int i = 0; i < user_feature[(int)personalized::HISTORY].size(); i++) {
    user_history.push_back(string(user_feature[(int)personalized::HISTORY][i].first.pcData,
      user_feature[(int)personalized::HISTORY][i].first.cbData));
  }
  //load time feature
  this->LoadTimeFeature(time_t_string, time_feature);

  vector<SizedStringBuilder > res_feature_vec;
  //string name=lc+"_"+vid+".txt";
  //ProfilerStart(name.c_str());

  //uint64 t1,t3;
  //t1 = clock();
  //for (int i = 0; i < 3000; i++) {
    vector<string> ctr_item;
    string tmp_vid;
    CTR_feature ctr;
    CTR_feature* ctr_p = NULL;
    if (!vid_ctr.empty()) {
      ctr.Load(vid_ctr, tmp_vid);
      ctr_p = &ctr;
    }
    this->LoadItemFeatureFromThrift(&media_doc_info, ctr_p, NULL, item_feature, now);
    this->FeatureFromRawFeatureBuild(features, res_feature_vec);
  //}
  //t3 = clock();
  //LOG(INFO) << (double)(t3 - t1)/CLOCKS_PER_SEC  << "ms " << res_feature_vec.size() << endl;
  //ProfilerStop();

  vector<string> res;
  std::ostringstream res_str;
  for (int i = 0;i < user_history.size(); i++) {
    res_str << user_history[i] << "|";
  }
  res_str << "\t";

  for (int i = 0; i < res_feature_vec.size(); i++) {
    res_str << res_feature_vec[i].str();
    res_str << ":" << res_feature_vec[i].weight;
    if (i != res_feature_vec.size() -1) {
      res_str << "\t";
    }
  }
  //res_str << res_feature_vec.size();
  return  res_str.str();



}


/*
currently contain the feature as below
  user profile:
    category, subcategory, tag, leword, people
  video attribute:
    category, subcategory, tag, leword, people, pid, ctr(rate_value, rate_log, rate_buckt, click_value, click_log, click_buckt, exp_value, exp_log, exp_buck), date (release, create), dura
*/

void FeatureParser(const string& line,
  std::vector<std::pair<const std::string*, float> >& tmp_feature_list,
  vector<string>& string_buffer) {

  std::vector<string> tmp_vec;
  SplitString(line, '\t', &tmp_vec);
  for (int i = 0; i < tmp_vec.size(); i++)
  {
      if (tmp_vec[i].size() == 0) {
          continue;
      }
      vector<string> kv_pair;
      SplitString(tmp_vec[i], ':', &kv_pair);
      CHECK(kv_pair.size() == 2);
      double tmp_double = -1.0;
      CHECK(StringToDouble(kv_pair[1], &tmp_double));
      string_buffer.push_back(kv_pair[0]);
      tmp_feature_list.push_back(make_pair(&string_buffer.back(), tmp_double));
  }
}

string GeneralRecommenderFeatureBuilder::PythonUtil::BuildFeature(
    const string& lc,
    const string& user_profile_string,
    const string& vid,
    const string& media_doc_info_string,
    const std::string& vid_ctr,
    const std::string& time_t_string) {

  //construct feature v
  Arena arena;
  GeneralRecommenderFeatureBuilder builder(true, &arena);
  std::vector<std::vector<std::pair<SizedString, float> > > user_feature((int)personalized::FEATURN_COUNT);
  std::vector<std::vector<std::pair<SizedString, float> > > item_feature((int)personalized::FEATURN_COUNT);
  std::vector<std::vector<std::pair<SizedString, float> > > time_feature(TimeFeatureTypeNum);
  std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* > features(FeatureSourceNum);
  features[FromUser]=&user_feature;
  features[FromItem]=&item_feature;
  features[FromTime]=&time_feature;

  time_t now;
  CHECK(StringToInt64(time_t_string, &now)) << time_t_string;

  //deserialization user profile
  string user_profile_string_base64 = user_profile_string;

  util::Base64Encoder encoder;
  string info_line("");
  if (!encoder.Decode(user_profile_string_base64, &info_line)){
    CHECK(false);
  }
  profile::UserProfile user_profile;
  if (!base::FromStringToThrift(info_line, &user_profile)) {
    CHECK(false);
  }

  //deserialization media doc info
  string media_doc_info_string_base64 = media_doc_info_string;
  string media_info_line("");
  if (!encoder.Decode(media_doc_info_string_base64, &media_info_line)){
    CHECK(false);
  }
  serving::MediaDocInfo media_doc_info;
  if (!base::FromStringToThrift(media_info_line, &media_doc_info)) {
    CHECK(false);
  }

  //load user profile
  builder.LoadUserProfileFromThrift(&user_profile, NULL, user_feature);
  // get user profile
  std::vector<std::string> user_history;
  for (int i = 0; i < user_feature[(int)personalized::HISTORY].size(); i++) {
    user_history.push_back(string(user_feature[(int)personalized::HISTORY][i].first.pcData,
      user_feature[(int)personalized::HISTORY][i].first.cbData));
  }
  //load time feature
  builder.LoadTimeFeature(time_t_string, time_feature);

  vector<SizedStringBuilder > res_feature_vec;
  //string name=lc+"_"+vid+".txt";
  //ProfilerStart(name.c_str());

  //uint64 t1,t3;
  //t1 = clock();
  //for (int i = 0; i < 3000; i++) {
    vector<string> ctr_item;
    string tmp_vid;
    CTR_feature ctr;
    CTR_feature* ctr_p = NULL;
    if (!vid_ctr.empty()) {
      ctr.Load(vid_ctr, tmp_vid);
      ctr_p = &ctr;
    }
    builder.LoadItemFeatureFromThrift(&media_doc_info, ctr_p, NULL, item_feature, now);
    builder.FeatureFromRawFeatureBuild(features, res_feature_vec);
  //}
  //t3 = clock();
  //LOG(INFO) << (double)(t3 - t1)/CLOCKS_PER_SEC  << "ms " << res_feature_vec.size() << endl;
  //ProfilerStop();

  vector<string> res;
  std::ostringstream res_str;
  for (int i = 0;i < user_history.size(); i++) {
    res_str << user_history[i] << "|";
  }
  res_str << "\t";

  for (int i = 0; i < res_feature_vec.size(); i++) {
    res_str << res_feature_vec[i].str();
    res_str << ":" << res_feature_vec[i].weight;
    if (i != res_feature_vec.size() -1) {
      res_str << "\t";
    }
  }
  //res_str << res_feature_vec.size();
  return  res_str.str();
}

std::string GeneralRecommenderFeatureBuilder::PythonUtil::test(std::vector<string>& list) {
  string res;
  for (int i = 0; i < list.size(); i++) {
      res = res + list[i];
  }
  return res;
}


}

