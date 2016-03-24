// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "ctr_predict_feature_builder.h"
#include "shared/serving/manager/repository.h"


namespace personalized {

/*enum HappyToSeeFeatureType {
  RELEASE_TIME = shared::FeatureType::FEATURE_UNKNOWN + 0,
  CREATE_TIME = shared::FeatureType::FEATURE_UNKNOWN + 1,
  HISTORY = shared::FeatureType::FEATURE_UNKNOWN + 2,
  ItemCF = shared::FeatureType::FEATURE_UNKNOWN + 3
};*/

class HappyToSeeRecommenderFeatureBuilder
  : public CTRPredictFeatureBuilder {

public:
  HappyToSeeRecommenderFeatureBuilder(bool for_training, Arena* arena);
  virtual ~HappyToSeeRecommenderFeatureBuilder() {}

  virtual void CustomizedFeatureBuild(
    std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
    std::vector<SizedStringBuilder >& res_feature_vec);

  virtual void LoadItemFeatureFromResultMedia(
    const std::string& id,
    const serving::ResultMedia* media_doc_info,
    const CTR_feature* ctr_feature_p,
    Classifier* item_feature_set,
    //FeatureData* item_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& item_features_,
    const uint64& now);

  virtual std::string BuildFeatureOffline(
    const std::string& lc,
    const std::string& user_profile_string,
    const std::string& vid,
    const std::string& media_doc_info_string,
    const std::string& media_doc_info_pid_string,
    const std::string& vid_ctr,
    const std::string& time_t_string,
    std::vector<std::string>& history);

};


} // namespace personalized

