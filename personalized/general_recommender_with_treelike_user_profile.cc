// Copyright 2016 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include <string>
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/media_doc_info_types.h"
#include "shared/personalized/general_recommender_with_treelike_user_profile.h"
#include "recommendation/common/toft/compress/block/block_compression.h"


using namespace toft;
using namespace std;

namespace personalized {

GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(bool for_training,
  Arena* arena): CTRPredictFeatureBuilder(for_training, arena){
}

void GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::CustomizedFeatureBuild(
  std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
  std::vector<SizedStringBuilder >& res_feature_vec) {
  SameTagFeature(features, res_feature_vec);
}

bool TagVectorSort(const std::pair<SizedString, float>& a, const std::pair<SizedString, float>& b) {
  return a.second > b.second;
}

void GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::LoadUserProfileFromThrift(const profile::UserProfile* user_profile,
    Classifier* user_feature_set,
    //FeatureData* user_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& user_features_) {

  if (user_profile == NULL) {
    VLOG(2) << "user profile construct is empty";
    return;
  }
  if (!user_profile->__isset.user_classify_interest) {
    VLOG(2) << "tag user feature is empty : " << user_profile->features.size() << " : " << user_profile->history.history_items.size();
    return;
  } else {
    VLOG(2) << "tag user feature not empty";
  }
  for (int i = 0; i < user_features_.size(); ++i) {
    user_features_[i].clear();
  }

  char buffer[1024];
  size_t len;
  char *p;
  SizedString tmp_str;
  // load the tag features
  // 1) build the category feature
  std::map<const std::string*, double> category_distribution;
  std::map<const std::string*, double>::iterator iter_category;
  double sum = 0.0;
  for (int i = 0; i < user_profile->user_classify_interest.cat_fea_data.size(); ++i) {
    category_distribution[&(user_profile->user_classify_interest.cat_fea_data[i].data.feature)] =
    user_profile->user_classify_interest.cat_fea_data[i].data.weight;
    sum = sum + user_profile->user_classify_interest.cat_fea_data[i].data.weight;
  }
  for (iter_category = category_distribution.begin(); iter_category != category_distribution.end(); ++iter_category) {
    float weight = iter_category->second / sum;
    size_t start = GetStartOfString(*(iter_category->first));
    SizedString(*(iter_category->first), start);
    if (user_feature_set == NULL || user_feature_set->IsContainUser(tmp_str)) {
      user_features_[(int)shared::FeatureType::CATEGORY].push_back(make_pair(tmp_str, weight));
    }
  }
  // 2) builder the tree tag feature
  for (int i = 0; i < user_profile->user_classify_interest.cat_fea_data.size(); ++i) {
    for (int j = 0; j < user_profile->user_classify_interest.cat_fea_data[i].children_cat_data.size(); ++j) {
      len = sprintf(buffer, "%d_%s", user_profile->user_classify_interest.cat_fea_data[i].data.feature_id,
      user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature.c_str());
      COPY_BUFFER(arena_, p, tmp_str, buffer, len);
      user_features_[(int)user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature_type].push_back(make_pair(tmp_str,
      user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].weight));

      if (user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature_type ==
        shared::FeatureType::TAG) {
        user_features_[shared::FeatureType::LABEL].push_back(make_pair(tmp_str, user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].weight));
      }
    }
  }
  // sort and cut the tag
  if (user_features_[shared::FeatureType::LABEL].size() > TagLimit) {
    std::sort(user_features_[shared::FeatureType::LABEL].begin(), user_features_[shared::FeatureType::LABEL].end(),
    TagVectorSort);
    user_features_[shared::FeatureType::LABEL].resize(TagLimit);
  }

  // add the gender, age and edu
  len = sprintf(buffer, "gender_%d", user_profile->demography_tag.gender_data.gender);
  COPY_BUFFER(arena_, p, tmp_str, buffer, len) ;
  user_features_[(int)personalized::GENDER].push_back(make_pair(tmp_str, user_profile->demography_tag.gender_data.weight));
  len = sprintf(buffer, "gender_%d", profile::Gender::MALE + profile::Gender::FEMALE - user_profile->demography_tag.gender_data.gender);
  COPY_BUFFER(arena_, p, tmp_str, buffer, len);
  user_features_[(int)personalized::GENDER].push_back(make_pair(tmp_str, 1 - user_profile->demography_tag.gender_data.weight));

  for (int it = 0; it < user_profile->demography_tag.age_data.size(); ++it) {
    len = sprintf(buffer, "age_%d", user_profile->demography_tag.age_data[it].age);
    COPY_BUFFER(arena_, p, tmp_str, buffer, len);
    user_features_[(int)personalized::AGE].push_back(make_pair(tmp_str, user_profile->demography_tag.age_data[it].weight));
  }

  for (int it = 0; it < user_profile->demography_tag.edu_data.size(); ++it) {
    len = sprintf(buffer, "edu_%d", user_profile->demography_tag.edu_data[it].education);
    COPY_BUFFER(arena_, p, tmp_str, buffer, len);
    user_features_[(int)personalized::EDU].push_back(make_pair(tmp_str, user_profile->demography_tag.edu_data[it].weight));
  }

  // get user history and album follow
  std::map<std::string, profile::UserHistory>::const_iterator it;
  std::map<const std::string, size_t> album_follow;
  for (it = user_profile->user_history.begin(); it != user_profile->user_history.end();
  ++it) {
    for (size_t i = 0; i < it->second.history_items.size(); ++i) {
      if (album_follow.find(it->second.history_items[i].id) != album_follow.end()) {
        album_follow[it->second.history_items[i].id]++;
      } else {
        album_follow[it->second.history_items[i].id] = 1;
      }
      size_t pos_prefix = it->second.history_items[i].id.find_first_of('_');
      if (pos_prefix == string::npos) {
        pos_prefix = 0;
      } else {
        pos_prefix++;
      }
      SizedString tmp_str(it->second.history_items[i].id, pos_prefix);
      user_features_[(int)personalized::HISTORY].push_back(make_pair(tmp_str, 1.0f));
    }
  }

  std::map<std::string, size_t>::iterator iter_album_follow;
  for (iter_album_follow = album_follow.begin(); iter_album_follow != album_follow.end(); ++iter_album_follow) {
    len = sprintf(buffer, "%s_%d", iter_album_follow->first.c_str(), int(ExponentDiscretize(iter_album_follow->second)));
    COPY_BUFFER(arena_, p, tmp_str, buffer, len);
    user_features_[(int)personalized::ALBUM_FOLLOW].push_back(make_pair(tmp_str, 1.0f));
  }
}

size_t GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::ExponentDiscretize(size_t num) {
  size_t exp = 1;
  size_t threshold = 2;
  while(exp < 6) {
    if (num < threshold) {
      break;
    }
    exp++;
    threshold = threshold << 1;
  }
  return exp;
}

void GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::LoadItemFeatureFromThriftWithPid(
    const serving::MediaDocInfo* media_doc_info,
    const serving::MediaDocInfo* media_doc_info_pid,
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

  // add the vedio attribution
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
        //item_feature_set->find(tmp _str) != item_feature_set->end()) {
        item_feature_set->IsContainItem(tmp_str)) {
        item_features_[iter->second.type].push_back(
          make_pair(tmp_str, 1.0f));
      }
    }
  }

  // add the album attribution
  if (media_doc_info_pid != NULL && !media_doc_info_pid->id.empty()) {
    std::map<std::string, shared::ItemFeature>::const_iterator iter_album;
    for (iter_album = media_doc_info_pid->feature_list.begin(); iter_album !=
      media_doc_info_pid->feature_list.end(); ++iter_album) {
      if (iter_album->second.type == shared::FeatureType::AREA) {
        size_t start = GetStartOfString(iter_album->first);
        if (start < iter_album->first.size()) {
          SizedString tmp_str(iter_album->first, start);
          if (item_feature_set == NULL ||
            //item_feature_set->find(tmp _str) != item_feature_set->end()) {
            item_feature_set->IsContainItem(tmp_str)) {
            item_features_[iter_album->second.type].push_back(
              make_pair(tmp_str, 1.0f));
          }
        }
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


string GeneralRecommenderWithTreelikeUserprofileFeatureBuilder::BuildFeatureOffline(
    const string& lc,
    const string& user_profile_string,
    const string& vid,
    const string& media_doc_info_string,
    const string& media_doc_info_pid_string,
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

  string uncompressed;
  BlockCompression* compression_ = TOFT_CREATE_BLOCK_COMPRESSION("snappy");
  if (!compression_->Uncompress(info_line.c_str(), info_line.size(), &uncompressed)){
     //std::err<< "uncompressed failed" << std::endl;
     CHECK(false);
  }
  delete compression_;

  profile::UserProfile user_profile;
  if (!base::FromStringToThrift(uncompressed, &user_profile)) {
    CHECK(false);
  }

  //load user profile
  this->LoadUserProfileFromThrift(&user_profile, NULL, user_feature);
  if (!user_profile.__isset.user_classify_interest) {
    return string();
  }
  // get user profile
  for (int i = 0; i < user_feature[(int)personalized::HISTORY].size(); i++) {
    user_history.push_back(string(user_feature[(int)personalized::HISTORY][i].first.pcData,
      user_feature[(int)personalized::HISTORY][i].first.cbData));
  }
  //load time feature
  this->LoadTimeFeature(time_t_string, time_feature);
  vector<SizedStringBuilder > res_feature_vec;
  vector<string> ctr_item;
  string tmp_vid;
  CTR_feature ctr;
  CTR_feature* ctr_p = NULL;
  if (!vid_ctr.empty()) {
    ctr.Load(vid_ctr, tmp_vid);
    ctr_p = &ctr;
  }
  // decode and deserialize media_doc_info
  string base64_decode_buffer;
  serving::MediaDocInfo vid_media_doc_info;
  serving::MediaDocInfo pid_media_doc_info;
  serving::MediaDocInfo* pid_media_doc_info_p = NULL;
  if (!encoder.Decode(media_doc_info_string, &base64_decode_buffer)) {
    CHECK(false);
  }
  if (!base::FromStringToThrift(base64_decode_buffer, &vid_media_doc_info)) {
    CHECK(false);
  }
  if (!media_doc_info_pid_string.empty()) {
    base64_decode_buffer.clear();
    if (!encoder.Decode(media_doc_info_pid_string, &base64_decode_buffer)) {
      CHECK(false);
    }
    if (!base::FromStringToThrift(base64_decode_buffer, &pid_media_doc_info)) {
      CHECK(false);
    }
    pid_media_doc_info_p = &pid_media_doc_info;
  }
  this->LoadItemFeatureFromThriftWithPid(&vid_media_doc_info, pid_media_doc_info_p, ctr_p, NULL, item_feature, now);
  this->FeatureFromRawFeatureBuild(features, res_feature_vec);

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


} // namespace personalized
