// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include <iostream>
#include "ctr_predict_feature_builder.h"
#include "happy_to_see_recommender_feature_builder.h"
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "util/global_init/global_init.h"
#include "util/encode/base64encoder.h"
#include "shared/serving/manager/repository.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/media_doc_info_types.h"



using namespace std;

namespace personalized {

HappyToSeeRecommenderFeatureBuilder::HappyToSeeRecommenderFeatureBuilder(bool for_training, Arena* arena)
  :CTRPredictFeatureBuilder(for_training_, arena){

  for_training_ = for_training;
  arena_ = arena;

  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::PID, Bool, FromItem));
  //rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::TAG, Bool, FromItem));
  //rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::LEWORD, Bool, FromItem));
  //rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::CATEGORY, Bool, FromItem));
  //rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  //rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::PEOPLE, Bool, FromItem));
  // pre compute in model
  rules_.push_back(new FeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));

  // history
  /*rules_.push_back(new FeatureBuildRule(2, personalized::HISTORY, Bool, FromUser,
                                       shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::HISTORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));*/

  // ctr feature
  /*rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::CTR, Float, FromItem,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));*/
  // intersection tag
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::LEWORD, Bool, FromItem));*/
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                     shared::FeatureType::PID, Bool, FromItem));
  // intersection subcategory
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));
  /*rules_.push_back(new FeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
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
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));*/
  /*rules_.push_back(new FeatureBuildRule(3, shared::FeatureType::CATEGORY, Bool, FromUser,
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

  rules_.push_back(new FeatureBuildRule(2, personalized::GENDER, Float, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::GENDER, Float, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::EDU, Float, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new FeatureBuildRule(2, personalized::EDU, Float, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));

}

void HappyToSeeRecommenderFeatureBuilder::CustomizedFeatureBuild(
  std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
  std::vector<SizedStringBuilder >& res_feature_vec) {

  SameTagFeature(features, res_feature_vec);
}

void HappyToSeeRecommenderFeatureBuilder::LoadItemFeatureFromResultMedia(
    const std::string& id,
    const serving::ResultMedia* result_media,
    const CTR_feature* ctr_feature_p,
    Classifier* item_feature_set,
    //FeatureData* item_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
    const uint64& now) {
  char buffer[BUFFER_SIZE];
  char* p;
  SizedString tmp_str;
  if (result_media == NULL || result_media->feature_list.size() == 0) {
    return;
  }
  for (int i = 0; i < item_features_.size(); ++i) {
    item_features_[i].clear();
  }

  if (!result_media->pid.empty())
  {
    COPY_BUFFER(arena_, p, tmp_str, result_media->pid.c_str(), result_media->pid.size());
    item_features_[(int)shared::FeatureType::PID].push_back(
      make_pair(tmp_str, 1.0f));
  }
  /*COPY_BUFFER(arena_, p, tmp_str, result_media->cid.c_str(), result_media->cid.size());
  item_features_[(int)shared::FeatureType::CATEGORY].push_back(
      make_pair(tmp_str, 1.0f));*/
  std::vector<serving::Feature>::const_iterator iter;
  for (iter = result_media->feature_list.begin();
       iter != result_media->feature_list.end(); ++iter) {

    size_t start = GetStartOfString(iter->name);
    if (start < iter->name.size()) {
      COPY_BUFFER(arena_, p, tmp_str, iter->name.c_str()+start, iter->name.size() - start);
      if (item_feature_set == NULL ||
        //item_feature_set->find(tmp_str) != item_feature_set->end()) {
        item_feature_set->IsContainItem(tmp_str)) {
        item_features_[iter->feature_type].push_back(
          make_pair(tmp_str, iter->weight));
      }
    }
  }

  size_t len;
  //const char* p;
  char* mem_p;
  //SizedString tmp_str;
  //prepare the ctr feature
  vector<pair<SizedString, float> >& item_ctr_features =
      item_features_[shared::FeatureType::CTR];
  if (ctr_feature_p != NULL) {

    int rate_int = static_cast<int>(ctr_feature_p->rate * 20.0f);
    int click_int = log(ctr_feature_p->click + 1) / log(2);
    int exp_int = log(ctr_feature_p->exp + 1) / log(2);

    len = sprintf(buffer, "rate%d", rate_int);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, item_ctr_features, 1.0f)
    len = sprintf(buffer, "click%d", click_int);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, item_ctr_features, 1.0f)
    len = sprintf(buffer, "rate%d-exp%d", rate_int, exp_int);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, item_ctr_features, 1.0f)

    ADD_CONST_STRING(tmp_str, rate_value, rate_value_len, item_ctr_features, ctr_feature_p->rate_value)
    len = sprintf(buffer, "rate_buckt_%d", ctr_feature_p->rate_buckt);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, item_ctr_features, 1.0f)

    /*ADD_CONST_STRING(tmp_str, click_value, click_value_len, item_ctr_features, ctr_feature_p->click_value)
    len = sprintf(buffer, "click_buckt_%d", ctr_feature_p->click_buckt);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, item_ctr_features, 1.0f)*/

    ADD_CONST_STRING(tmp_str, y_recvid_rate, y_recvid_rate_len, item_ctr_features, 1.0f)
  }
  else {
    ADD_CONST_STRING(tmp_str, n_recvid_rate, n_recvid_rate_len, item_ctr_features, 1.0f)
  }

  SizedString id_str(id);
  item_features_[shared::FeatureType::VID].push_back(
      make_pair(id_str, 1.0f));

  //prepare release time and create time feature
  time_t release_time_t, create_time_t;
  release_time_t = result_media->release_timestamp;
  create_time_t = result_media->create_timestamp;

  struct tm* release_tm;
  vector<pair<SizedString, float> >& release_time_features =
    item_features_[personalized::RELEASE_TIME];
  if (release_time_t != 0) {
    uint64 diff_time_t = now - release_time_t;
    if (diff_time_t < 2592000) {
      ADD_CONST_STRING(tmp_str, release_month_1, release_month_1_len, release_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 2) {
      ADD_CONST_STRING(tmp_str, release_month_2, release_month_2_len, release_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 3) {
      ADD_CONST_STRING(tmp_str, release_month_3, release_month_3_len, release_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 6) {
      ADD_CONST_STRING(tmp_str, release_month_6, release_month_6_len, release_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 24) {
      ADD_CONST_STRING(tmp_str, release_month_24, release_month_24_len, release_time_features, 1.0f)
    } else {
      ADD_CONST_STRING(tmp_str, release_month_old, release_month_old_len, release_time_features, 1.0f)
    }
    release_tm=localtime(&release_time_t);
    len = sprintf(buffer, "release_time_%d", (release_tm->tm_year%100)/10);
    ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, release_time_features, 1.0f)
  } else {
    ADD_CONST_STRING(tmp_str, release_null, release_null_len, release_time_features, 1.0f)
  }

  vector<pair<SizedString, float> >& create_time_features =
    item_features_[personalized::CREATE_TIME];

  if (create_time_t != 0) {
    uint64 diff_time_t = now - create_time_t;
    if (diff_time_t < 86400) {
      ADD_CONST_STRING(tmp_str, create_day_1, create_day_1_len, create_time_features, 1.0f)
    } else if (diff_time_t < 86400 * 2) {
      ADD_CONST_STRING(tmp_str, create_day_2, create_day_2_len, create_time_features, 1.0f)
    } else if (diff_time_t < 2592000) {
      ADD_CONST_STRING(tmp_str, create_month_1, create_month_1_len, create_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 2) {
      ADD_CONST_STRING(tmp_str, create_month_2, create_month_2_len, create_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 3) {
      ADD_CONST_STRING(tmp_str, create_month_3, create_month_3_len, create_time_features, 1.0f)
    } else if (diff_time_t < 2592000 * 6) {
      ADD_CONST_STRING(tmp_str, create_month_6, create_month_6_len, create_time_features, 1.0f)
    } else {
      ADD_CONST_STRING(tmp_str, create_month_old, create_month_old_len, create_time_features, 1.0f)
    }
  } else {
      ADD_CONST_STRING(tmp_str, create_null, create_null_len, create_time_features, 1.0f)
  }

}

std::string HappyToSeeRecommenderFeatureBuilder::BuildFeatureOffline(
  const std::string& lc,
  const std::string& user_profile_string,
  const std::string& vid,
  const std::string& media_doc_info_string,
  const std::string& media_doc_info_pid_string,
  const std::string& vid_ctr,
  const std::string& time_t_string,
  std::vector<std::string>& user_history) {

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
  //load user profile
  LoadUserProfileFromThrift(&user_profile, NULL, user_feature);
  // get user profile
  for (int i = 0; i < user_feature[(int)personalized::HISTORY].size(); i++) {
    user_history.push_back(string(user_feature[(int)personalized::HISTORY][i].first.pcData,
      user_feature[(int)personalized::HISTORY][i].first.cbData));
  }

  //deserialization media doc info
  string media_doc_info_string_base64 = media_doc_info_string;
  string media_info_line("");
  if (!encoder.Decode(media_doc_info_string_base64, &media_info_line)){
    CHECK(false);
  }
  serving::MediaDocInfo media_doc_info;
  serving::ResultMedia result_media;
  if (!base::FromStringToThrift(media_info_line, &media_doc_info)) {
    CHECK(false);
  }
  serving::ResultMedia::ConvertMediaDocToResultMedia(media_doc_info, &result_media);
  //load time feature
  LoadTimeFeature(time_t_string, time_feature);

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
    LoadItemFeatureFromResultMedia(vid, &result_media, ctr_p, NULL, item_feature, now);
    FeatureFromRawFeatureBuild(features, res_feature_vec);
  //}
  //t3 = clock();
  //LOG(INFO) << (double)(t3 - t1)/CLOCKS_PER_SEC  << "ms " << res_feature_vec.size() << endl;
  //ProfilerStop();

  vector<string> res;
  std::ostringstream res_str;

  for (int i = 0; i < res_feature_vec.size(); i++) {
    res_str << res_feature_vec[i].str();
    res_str << ":" << res_feature_vec[i].weight;

    if (i != res_feature_vec.size() -1) {
      res_str << "\t";
    }
  }
  string feature = res_str.str();
  return feature;
}

} // namespace personalized


