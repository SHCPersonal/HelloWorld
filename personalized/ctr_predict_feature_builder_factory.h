// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "string"
#include "shared/personalized/ctr_predict_feature_builder.h"
#include "shared/personalized/general_recommender_feature_builder.h"
#include "shared/personalized/general_recommender_with_treelike_user_profile.h"
#include "shared/personalized/general_recommender_with_treelike_user_profile_pro.h"
#include "shared/personalized/sarrs_extra_feature_builder.h"


namespace personalized {

class CTRPredictFeatureBuilderFactory {
public:
  static CTRPredictFeatureBuilder * GetInstance(const std::string& name, bool training, Arena* arena) {
    if (name == "GeneralRecommenderFeatureBuilder") {
      return new GeneralRecommenderFeatureBuilder(training, arena);
    } else if (name == "HappyToSeeRecommenderFeatureBuilder") {
      return  new HappyToSeeRecommenderFeatureBuilder(training, arena);
    } else if (name == "GeneralRecommender1") {
      return new GeneralRecommender1(training, arena);
    } else if (name == "GeneralRecommender2") {
      return new GeneralRecommender2(training, arena);
    } else if (name == "GeneralRecommender3") {
      return new GeneralRecommender3(training, arena);
    } else if (name == "GeneralRecommender4") {
      return new GeneralRecommender4(training, arena);
    } else if (name == "GeneralRecommender5") {
      return new GeneralRecommender5(training, arena);
    } else if (name == "SarrsExtraLabelFeatureBuilder") {
      return new SarrsExtraLabelFeatureBuilder(training, arena);
    } else if (name == "SarrsExtraQueryFeatureBuilder") {
      return new SarrsExtraQueryFeatureBuilder(training, arena);
    } else if (name == "GeneralRecommenderPro1") {
      return new GeneralRecommenderPro1(training, arena);
    } else if (name == "GeneralRecommenderPro4") {
      return new GeneralRecommenderPro4(training, arena);
    } else if (name == "GeneralRecommenderPro5") {
      return new GeneralRecommenderPro5(training, arena);
    } else {
      CHECK(false);
    }
    return NULL;
  }
};

} // namespace personalized
