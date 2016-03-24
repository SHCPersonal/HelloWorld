// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "ctr_predict_feature_builder.h"

namespace personalized {

/*enum ExtraFeatureType {
  ItemCF = shared::FeatureType::FEATURE_UNKNOWN + 0
};*/

class GeneralRecommenderFeatureBuilder
  : public CTRPredictFeatureBuilder {
public:
  GeneralRecommenderFeatureBuilder(bool for_training, Arena* arena);
  virtual ~GeneralRecommenderFeatureBuilder() {};

  virtual void CustomizedFeatureBuild(
    std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
    std::vector<SizedStringBuilder >& res_feature_vec);

  virtual std::string BuildFeatureOffline(
    const std::string& lc,
    const std::string& user_profile_string,
    const std::string& vid,
    const std::string& media_doc_info_string,
    const std::string& media_doc_info_pid_string,
    const std::string& vid_ctr,
    const std::string& time_t_string,
    std::vector<std::string>& history);

  class PythonUtil {
  public:
    std::string BuildFeature(
    const std::string& lc,
    const std::string& user_profile_string,
    const std::string& vid,
    const std::string& media_doc_info_string,
    const std::string& vid_ctr,
    const std::string& time_t_string);

    std::string test(std::vector<std::string>& list);
  };
};

}
