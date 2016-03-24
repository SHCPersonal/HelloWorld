// Copyright 2016 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "general_recommender_with_treelike_user_profile.h"

namespace personalized {

class GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:

  GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder(bool for_training,
    Arena* arena);
  virtual ~GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder() {};

  virtual void LoadUserProfileFromThrift(
    const profile::UserProfile* user_profile,
    Classifier* user_feature_set,
    //FeatureData* user_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& user_features_);


protected:

};


class GeneralRecommenderPro1 :
  public GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder {

public:
  GeneralRecommenderPro1(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder(for_training_,
    arena) {
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::TAG, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::LEWORD, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::SUBCATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::PEOPLE, Bool, FromItem));
    // pre compute in model
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));

    // inter album follow
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::ALBUM_FOLLOW, Bool, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));

    // inter area and pid
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Bool, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));

    // inter area
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Bool, FromUser,
                                         shared::FeatureType::AREA, Bool, FromItem));
    // intersection tag
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::LABEL, Bool, FromUser,
                                         shared::FeatureType::TAG, Bool, FromItem));
    // intersection subcategory
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));

    // release time and create time
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, personalized::RELEASE_TIME, Bool, FromItem));

    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
  }
  virtual ~GeneralRecommenderPro1(){}

};

class GeneralRecommenderPro4 :
  public GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder {

public:
  GeneralRecommenderPro4(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder(for_training_,
    arena) {
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::TAG, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::LEWORD, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::SUBCATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::PEOPLE, Bool, FromItem));
    // pre compute in model
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));

    // inter album follow
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::ALBUM_FOLLOW, Float, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));
    // inter area and pid
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Float, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));

    // inter area
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Float, FromUser,
                                         shared::FeatureType::AREA, Bool, FromItem));
    // intersection tag
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::LABEL, Float, FromUser,
                                         shared::FeatureType::TAG, Bool, FromItem));
    // intersection subcategory
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));


    // release time and create time
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, personalized::RELEASE_TIME, Bool, FromItem));

    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));

    // add the age and gender
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::GENDER, Float, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  }

  virtual ~GeneralRecommenderPro4() {};
};


class GeneralRecommenderPro5 :
  public GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder {

public:
  GeneralRecommenderPro5(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileProFeatureBuilder(for_training_,
    arena) {
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::VID, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::TAG, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::LEWORD, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::SUBCATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::PEOPLE, Bool, FromItem));
    // pre compute in model
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, shared::FeatureType::CTR, Float, FromItem));

    // inter album follow
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::ALBUM_FOLLOW, Float, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));
    // inter area and pid
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Float, FromUser,
                                         shared::FeatureType::PID, Bool, FromItem));

    // inter area
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Float, FromUser,
                                         shared::FeatureType::AREA, Bool, FromItem));
    // intersection tag
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::LABEL, Float, FromUser,
                                         shared::FeatureType::TAG, Bool, FromItem));
    // intersection subcategory
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));


    // release time and create time
    rules_.push_back(new PrefixMappingFeatureBuildRule(1, personalized::RELEASE_TIME, Bool, FromItem));

    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::CATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Bool, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
  }

  virtual ~GeneralRecommenderPro5() {};
};


} // namespace personalized
