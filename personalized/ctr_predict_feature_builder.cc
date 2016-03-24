// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "sized_string.h"
#include "ctr_predict_feature_builder.h"
#include <iostream>
#include <sstream>
#include "math.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/media_doc_info_types.h"
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "util/global_init/global_init.h"
#include "util/encode/base64encoder.h"
#include "third_party/gperftools-2.0/src/gperftools/profiler.h"

using namespace std;

namespace personalized {

DEFINE_STRING(rate_value)
//DEFINE_STRING(rate_log)
//DEFINE_STRING(click_value)
//DEFINE_STRING(click_log)
//DEFINE_STRING(exp_value)
//DEFINE_STRING(exp_log)
DEFINE_STRING(y_recvid_rate)
DEFINE_STRING(n_recvid_rate)
DEFINE_STRING(release_month_1)
DEFINE_STRING(release_month_2)
DEFINE_STRING(release_month_3)
DEFINE_STRING(release_month_6)
DEFINE_STRING(release_month_24)
DEFINE_STRING(release_month_old)
DEFINE_STRING(release_null)
DEFINE_STRING(create_day_1)
DEFINE_STRING(create_day_2)
DEFINE_STRING(create_month_1)
DEFINE_STRING(create_month_2)
DEFINE_STRING(create_month_3)
DEFINE_STRING(create_month_6)
DEFINE_STRING(create_month_old)
DEFINE_STRING(create_null)

const string same_tag_prefix = "sa_tag";
const char string_splitor_c = '|';
//extern const char string_splitor_c;

FeatureTypeMapping g_feature_mapping;

CTRPredictFeatureBuilder::~CTRPredictFeatureBuilder()
{
  for (size_t i = 0; i < rules_.size(); ++i) {
    delete rules_[i];
  }
}


CTRPredictFeatureBuilder::CTRPredictFeatureBuilder(bool for_training, Arena* arena)
  :arena_(arena),
   for_training_(for_training) {

}

void CTRPredictFeatureBuilder::FeatureFromRawFeatureBuild(
    std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
    vector<SizedStringBuilder >& res_feature_vec) {

  res_feature_vec.clear();
  res_feature_vec.reserve(2048);
  SizedString tmp_str;

  // call the customized feature builder
  this->CustomizedFeatureBuild(features, res_feature_vec);

  for (int i = 0; i < rules_.size(); ++i) {
    const FeatureBuildRule* rule = rules_[i];
    SizedString prefix_tmp_str(rule->prefix_str);
    switch (rule->combine_num) {
      case 1: {
        vector<pair<SizedString, float> >& features1 =
            (*features[rule->feature1_source])[rule->feature1_type];

        vector<pair<SizedString, float> >::iterator iter1 = features1.begin();
        for (; iter1 != features1.end(); ++iter1) {
          float score = 1.0f;
          if (rule->feature1_weight_type == Float) {
            score = iter1->second;
          }
          SingleFeatureGenerate(prefix_tmp_str, iter1->first ,score, res_feature_vec);
        }
        break;
      }
      case 2: {
        vector<pair<SizedString, float> >& features1 =
            (*features[rule->feature1_source])[rule->feature1_type];
        vector<pair<SizedString, float> >& features2 =
            (*features[rule->feature2_source])[rule->feature2_type];

        vector<pair<SizedString, float> >::iterator iter1;
        vector<pair<SizedString, float> >::iterator iter2;
        for (iter1 = features1.begin(); iter1 != features1.end(); ++iter1) {
          float score1 = 1.0f;
          if (rule->feature1_weight_type == Float) {
            score1 = iter1->second;
          }
          for (iter2 = features2.begin(); iter2 != features2.end(); ++iter2) {
            float score2 = 1.0f;
            if (rule->feature2_weight_type == Float) {
              score2 = iter2->second;
            }
            TwoFeatureGenerate(prefix_tmp_str, iter1->first, iter2->first, score1 * score2, res_feature_vec);
          }
        }
        break;
      }
      case 3: {
        vector<pair<SizedString, float> >& features1 =
            (*features[rule->feature1_source])[rule->feature1_type];
        vector<pair<SizedString, float> >& features2 =
            (*features[rule->feature2_source])[rule->feature2_type];
        vector<pair<SizedString, float> >& features3 =
            (*features[rule->feature3_source])[rule->feature3_type];

        vector<pair<SizedString, float> >::iterator iter1;
        vector<pair<SizedString, float> >::iterator iter2;
        vector<pair<SizedString, float> >::iterator iter3;
        for (iter1 = features1.begin(); iter1 != features1.end(); ++iter1) {
          float score1 = 1.0f;
          if (rule->feature1_weight_type == Float) {
            score1 = iter1->second;
          }
          for (iter2 = features2.begin(); iter2 != features2.end(); ++iter2) {
            float score2 = 1.0f;
            if (rule->feature2_weight_type == Float) {
              score2 = iter2->second;
            }
            for (iter3 = features3.begin(); iter3 != features3.end(); ++iter3) {
              float score3 = 1.0f;
              if (rule->feature3_weight_type == Float) {
                score3 = iter3->second;
              }
              ThreeFeatureGenerate(
                  prefix_tmp_str, iter1->first, iter2->first, iter3->first,
                  score1 * score2 * score3, res_feature_vec);
            }
          }
        }
        break;
      }
      default: {
        CHECK(false);
        break;
      }
    }
  }
  return;
}

int CompareSizedString(const SizedString& a, const SizedString& b) {
  size_t len = std::min(a.cbData, b.cbData);
  for (int i = 0; i < len; ++i) {
    if (a.pbData[i] < b.pbData[i]) {
      return -1;
    } else if(a.pbData[i] > b.pbData[i]) {
      return 1;
    }
  }
  if (a.cbData < b.cbData) {
    return -1;
  } else if (a.cbData > b.cbData) {
    return 1;
  } else {
    return 0;
  }
}

bool ComparePairFirst(const pair<SizedString, float>& l,
                      const pair<SizedString, float>& r) {
  if (CompareSizedString(l.first, r.first) < 0)
    return true;
  else
    return false;
}

void CTRPredictFeatureBuilder::SameTagFeature(std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
    std::vector<SizedStringBuilder >& res_feature_vec) {

  const char* p;
  size_t len;
  SizedString tmp_str;
  vector<pair<SizedString, float> >& user_tag_features =
      (*features[FromUser])[shared::FeatureType::TAG];
  vector<pair<SizedString, float> >& item_tag_features =
      (*features[FromItem])[shared::FeatureType::TAG];

  if (!user_tag_features.empty() && !item_tag_features.empty()) {
    sort(user_tag_features.begin(), user_tag_features.end(),
         ComparePairFirst);
    sort(item_tag_features.begin(), item_tag_features.end(),
         ComparePairFirst);
    float sim_f = GetSimAndSet(same_tag_prefix,
                               user_tag_features,
                               item_tag_features,
                               res_feature_vec);
    if (sim_f > 0.001) {
      p = "comtagsim";
      len = strlen(p);
      tmp_str.set(p, len);
      SizedString same_tag_prefix_str(same_tag_prefix);
      SingleFeatureGenerate(same_tag_prefix_str, tmp_str, sim_f, res_feature_vec);
    }
  }
}

float CTRPredictFeatureBuilder::GetSimAndSet(
    const std::string& prefix,
    vector<pair<SizedString, float> >& features1,
    vector<pair<SizedString, float> >& features2,
    std::vector<SizedStringBuilder >& res_feature_vec) {
  vector<pair<SizedString, float> >::iterator iter1 = features1.begin();
  vector<pair<SizedString, float> >::iterator iter2 = features2.begin();

  float inter_set_num = 0;
  float union_set_num = 0;
  SizedString prefix_tmp_str(prefix);
  while (iter1 != features1.end() && iter2 != features2.end()) {
    if (CompareSizedString(iter1->first,iter2->first) < 0) {
      ++iter1;
    } else if (CompareSizedString(iter1->first,iter2->first) > 0) {
      ++iter2;
    } else {
      SingleFeatureGenerate(prefix_tmp_str, iter1->first, 1.0f, res_feature_vec);
      ++inter_set_num;
      iter1++;
      iter2++;
    }
    ++union_set_num;
  }
  float res = inter_set_num / union_set_num;
  res = static_cast<int>((res * 1000) + 0.5f) / 1000.0f;
  return res;
}


size_t CTRPredictFeatureBuilder::GetStartOfString(const std::string& str) {
  size_t i;
  for (i = 0; i < str.size(); ++i) {
    if (str[i] == string_splitor_c) {
      return str.size();
    }
  }
  for (i = 0; i < str.size(); i++) {
    if (str[i] == '_') {
      break;
    }
  }
  if (i != str.size() && i != str.size() - 1) {
    return i + 1;
  }
  return 0;
}


void CTRPredictFeatureBuilder::SingleFeatureGenerate(
    SizedString& prefix,
    SizedString& feature_name,
    float weight,
    std::vector<SizedStringBuilder >& res_feature_vec) {
  res_feature_vec.push_back(SizedStringBuilder());
  SizedStringBuilder& tmp_builder = res_feature_vec.back();
  tmp_builder.size = 2;
  tmp_builder.builder[0] = prefix;
  tmp_builder.builder[1] = feature_name;
  tmp_builder.weight = weight;
}


void CTRPredictFeatureBuilder::TwoFeatureGenerate(
     SizedString& prefix,
     SizedString& feature_name1,
     SizedString& feature_name2,
     float weight,
     std::vector<SizedStringBuilder >& res_feature_vec) {
  res_feature_vec.push_back(SizedStringBuilder());
  SizedStringBuilder& tmp_builder = res_feature_vec.back();
  tmp_builder.size = 3;
  tmp_builder.builder[0] = prefix;
  tmp_builder.builder[1] = feature_name1;
  tmp_builder.builder[2] = feature_name2;
  tmp_builder.weight = weight;
}

void CTRPredictFeatureBuilder::ThreeFeatureGenerate(
     SizedString& prefix,
     SizedString& feature_name1,
     SizedString& feature_name2,
     SizedString& feature_name3,
     float weight,
     std::vector<SizedStringBuilder >& res_feature_vec) {
  res_feature_vec.push_back(SizedStringBuilder());
  SizedStringBuilder& tmp_builder = res_feature_vec.back();
  tmp_builder.size = 4;
  tmp_builder.builder[0] = prefix;
  tmp_builder.builder[1] = feature_name1;
  tmp_builder.builder[2] = feature_name2;
  tmp_builder.builder[3] = feature_name3;
  tmp_builder.weight = weight;
}


void CTRPredictFeatureBuilder::LoadUserProfileFromThrift(const profile::UserProfile* user_profile,
    Classifier* user_feature_set,
    //FeatureData* user_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& user_features_) {

  if (user_profile == NULL) {
    VLOG(2) << "user profile construct is empty";
    return;
  }
  if (!user_profile->__isset.features) {
    VLOG(2) << "tag user feature is empty : " << user_profile->features.size() << " : " << user_profile->history.history_items.size();
    return;
  } else {
    VLOG(2) << "tag user feature not empty";
  }
  for (int i = 0; i < user_features_.size(); ++i) {
    user_features_[i].clear();
  }
  // load the tag features
  map<string, profile::UserFeature>::const_iterator iter;
  for (iter = user_profile->features.begin();
       iter != user_profile->features.end(); ++iter) {
    float res = iter->second.weight / 1000.0f;
    if (res > 1.0) {
      res = 1.0;
    }

    size_t start = GetStartOfString(iter->first);
    if (start < iter->first.size()) {
      SizedString tmp_str(iter->first, start);
      if (user_feature_set == NULL ||
        //user_feature_set->find(tmp_str) != user_feature_set->end()) {
        user_feature_set->IsContainUser(tmp_str)) {
        user_features_[(int)iter->second.type].push_back(
             make_pair(tmp_str, res));
      }
    }
  }

  // get user history
  for (int i = 0; i < user_profile->history.history_items.size(); ++i) {
    size_t pos_prefix = user_profile->history.history_items[i].id.find_first_of('_');
    if (pos_prefix == string::npos) {
      pos_prefix = 0;
    } else {
      pos_prefix++;
    }
    SizedString tmp_str(user_profile->history.history_items[i].id, pos_prefix);
    user_features_[(int)personalized::HISTORY].push_back(make_pair(tmp_str, 1.0f));
  }

}

void CTRPredictFeatureBuilder::LoadItemFeatureFromThrift(
    const serving::MediaDocInfo* media_doc_info,
    const CTR_feature* ctr_feature_p,
    Classifier* item_feature_set,
    //FeatureData* item_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
    const uint64& now) {
  char buffer[BUFFER_SIZE];
  if (media_doc_info == NULL ||
      !media_doc_info->__isset.feature_list) {
    return;
  }
  for (int i = 0; i < item_features_.size(); ++i) {
    item_features_[i].clear();
  }

  SizedString pid_str(media_doc_info->pid);
  item_features_[(int)shared::FeatureType::PID].push_back(
      make_pair(pid_str, 1.0f));
  SizedString cid_str(media_doc_info->cid);
  item_features_[(int)shared::FeatureType::CATEGORY].push_back(
      make_pair(cid_str, 1.0f));
  std::map<std::string, shared::ItemFeature>::const_iterator iter;
  for (iter = media_doc_info->feature_list.begin();
       iter != media_doc_info->feature_list.end(); ++iter) {

    size_t start = GetStartOfString(iter->first);
    if (start < iter->first.size()) {
      SizedString tmp_str(iter->first, start);
      if (item_feature_set == NULL ||
        //item_feature_set->find(tmp_str) != item_feature_set->end()) {
        item_feature_set->IsContainItem(tmp_str)) {
        item_features_[iter->second.type].push_back(
          make_pair(tmp_str, 1.0f));
      }
    }

  }

  size_t len;
  //const char* p;
  char* mem_p;
  SizedString tmp_str;
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

  SizedString id_str(media_doc_info->id);
  item_features_[shared::FeatureType::VID].push_back(
      make_pair(id_str, 1.0f));

  //prepare release time and create time feature
  time_t release_time_t, create_time_t;
  release_time_t = media_doc_info->release_timestamp;
  create_time_t = media_doc_info->create_timestamp;

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
void CTRPredictFeatureBuilder::LoadTimeFeature(const std::string& time_str,
  std::vector<std::vector<std::pair<SizedString, float> > >& time_features_) {
  time_t t;
  CHECK(StringToInt64(time_str, &t)) << time_str;
  LoadTimeFeature(t, time_features_);
}

void CTRPredictFeatureBuilder::LoadTimeFeature(const time_t& t,
  std::vector<std::vector<std::pair<SizedString, float> > >& time_features_) {

  char buffer[BUFFER_SIZE];
  struct tm *local;
  local=localtime(&t);

  vector<pair<SizedString, float> >& small_time_features =
      time_features_[Time];

  SizedString tmp_str;
  char* mem_p;
  size_t len;
  /*len = sprintf(buffer, "h_%d", local->tm_hour);
  ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, small_time_features, 1.0f)*/
  len = sprintf(buffer, "hbkt_%d", local->tm_hour / 3);
  ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, small_time_features, 1.0f)
  /*len = sprintf(buffer, "wday_%d", local->tm_wday - 1);
  ADD_BUFFER(arena_, mem_p, tmp_str, buffer, len, small_time_features, 1.0f)*/
}

} //namespace personalized

