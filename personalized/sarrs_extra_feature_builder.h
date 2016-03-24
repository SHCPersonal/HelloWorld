// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "ctr_predict_feature_builder.h"
#include "shared/personalized/happy_to_see_recommender_feature_builder.h"
#include "shared/serving/manager/repository.h"


namespace personalized {

class SarrsExtraLabelFeatureBuilder : public HappyToSeeRecommenderFeatureBuilder {

public:
  SarrsExtraLabelFeatureBuilder(bool for_training, Arena* arena);
  virtual ~SarrsExtraLabelFeatureBuilder() {}
};

class SarrsExtraQueryFeatureBuilder : public HappyToSeeRecommenderFeatureBuilder {

public:
  SarrsExtraQueryFeatureBuilder(bool for_training, Arena* arena);
  virtual ~SarrsExtraQueryFeatureBuilder() {}
};


} // namespace personalized
