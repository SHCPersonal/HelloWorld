// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include <iostream>
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "util/global_init/global_init.h"
#include "util/encode/base64encoder.h"
#include "ctr_predict_feature_builder.h"
#include "shared/personalized/sarrs_extra_feature_builder.h"


using namespace std;

namespace personalized {

SarrsExtraLabelFeatureBuilder::SarrsExtraLabelFeatureBuilder(bool for_training, Arena* arena)
  :HappyToSeeRecommenderFeatureBuilder(for_training_, arena){

  for_training_ = for_training;
  arena_ = arena;

  //clear the rules in the parent class
  rules_.clear();

  // just want to use it to debug, as I know, it has none benefit for online ranking
  //rules_.push_back(FeatureBuildRule(1, shared::FeatureType::UID, Bool, FromUser));

  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::PID, Bool, FromItem));

  // pre compute in model
  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));


  // intersection tag
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::LABEL, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                     shared::FeatureType::PID, Bool, FromItem));
  // intersection subcategory
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::LABEL, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));
  // release time and create time
  rules_.push_back(new PrefixMappingFeatureBuildRule(1, personalized::RELEASE_TIME, Bool, FromItem));

  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                       personalized::RELEASE_TIME, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       personalized::RELEASE_TIME, Bool, FromItem));
}

SarrsExtraQueryFeatureBuilder::SarrsExtraQueryFeatureBuilder(bool for_training, Arena* arena)
  :HappyToSeeRecommenderFeatureBuilder(for_training_, arena){

  for_training_ = for_training;
  arena_ = arena;

  //clear the rules in the parent class
  rules_.clear();

  // just want to use it to debug, as I know, it has none benefit for online ranking
  //rules_.push_back(FeatureBuildRule(1, shared::FeatureType::VID, Bool, FromUser));

  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::PID, Bool, FromItem));
  // pre compute in model
  rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));
  // intersection tag
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::QUERY, Bool, FromItem));
  /*rules_.push_back(FeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                       shared::FeatureType::LEWORD, Bool, FromItem));*/
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::TAG, Bool, FromUser,
                                     shared::FeatureType::PID, Bool, FromItem));
  // intersection subcategory
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::TAG, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::QUERY, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                       shared::FeatureType::PID, Bool, FromItem));

}

} // namespace personalized

