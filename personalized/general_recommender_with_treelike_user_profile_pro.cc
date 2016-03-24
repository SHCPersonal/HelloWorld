// Copyright 2016 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma
#include "general_recommender_with_treelike_user_profile_pro.h"
#include <string>
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "shared/serving/proto/user_profile_types.h"
#include "shared/serving/proto/media_doc_info_types.h"
#include "shared/personalized/general_recommender_with_treelike_user_profile.h"
#include "recommendation/common/toft/compress/block/block_compression.h"


using namespace toft;
using namespace std;

namespace personalized {

GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder::GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder(bool for_training,
  Arena* arena): GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training, arena){
}

void GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder::LoadUserProfileFromThrift(const profile::UserProfile* user_profile,
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

      if (user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature_type ==
        shared::FeatureType::TAG) {
        user_features_[shared::FeatureType::LABEL].push_back(make_pair(tmp_str, user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].weight));
        tmp_str.set(user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature.c_str(),
        user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature.size());
        user_features_[shared::FeatureType::TAG].push_back(make_pair(tmp_str,
        user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].weight));
      } else {
        user_features_[(int)user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].feature_type].push_back(make_pair(tmp_str,
          user_profile->user_classify_interest.cat_fea_data[i].children_cat_data[j].weight));
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
      if (album_follow.find(it->second.history_items[i].pid) != album_follow.end()) {
        album_follow[it->second.history_items[i].pid]++;
      } else {
        album_follow[it->second.history_items[i].pid] = 1;
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


} // namespace personalized
