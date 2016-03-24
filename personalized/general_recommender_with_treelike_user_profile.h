// Copyright 2016 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include "ctr_predict_feature_builder.h"

#define TagLimit 100


namespace personalized {

bool TagVectorSort(const std::pair<SizedString, float>& a, const std::pair<SizedString, float>&
b);

class GeneralRecommenderWithTreelikeUserprofileFeatureBuilder :
  public CTRPredictFeatureBuilder {
public:
  GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(bool for_training,
  Arena* arena);
  virtual ~GeneralRecommenderWithTreelikeUserprofileFeatureBuilder() {};


  virtual void CustomizedFeatureBuild(std::vector<std::vector<std::vector<std::pair<SizedString, float> > >* >& features,
      std::vector<SizedStringBuilder >& res_feature_vec);

  virtual void LoadUserProfileFromThrift(
    const profile::UserProfile* user_profile,
    Classifier* user_feature_set,
    //FeatureData* user_feature_set,
    std::vector<std::vector<std::pair<SizedString, float> > >& user_features_);

  virtual size_t ExponentDiscretize(size_t num);

  virtual void LoadItemFeatureFromThriftWithPid(
    const serving::MediaDocInfo* media_doc_info,
    const serving::MediaDocInfo* media_doc_info_pid,
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
protected:

};

class GeneralRecommender1 :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:
  GeneralRecommender1(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training_,
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
  virtual ~GeneralRecommender1(){}

};

class GeneralRecommender2 :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:
  GeneralRecommender2(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training_,
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
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::AREA, Float, FromUser,
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

    // add the age and gender
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::AGE, Float, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, personalized::GENDER, Float, FromUser,
                                         shared::FeatureType::SUBCATEGORY, Bool, FromItem));
  }
  virtual ~GeneralRecommender2() {}
};


class GeneralRecommender3 :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:
  GeneralRecommender3(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training_,
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

    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::CATEGORY, Float, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
    rules_.push_back(new PrefixMappingFeatureBuildRule(2, shared::FeatureType::SUBCATEGORY, Float, FromUser,
                                         personalized::RELEASE_TIME, Bool, FromItem));
  }
  virtual ~GeneralRecommender3() {}

};

class GeneralRecommender4 :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:
  GeneralRecommender4(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training_,
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

  virtual ~GeneralRecommender4() {};
};


class GeneralRecommender5 :
  public GeneralRecommenderWithTreelikeUserprofileFeatureBuilder {

public:
  GeneralRecommender5(bool for_training, Arena* arena)
    :GeneralRecommenderWithTreelikeUserprofileFeatureBuilder(for_training_,
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

  virtual ~GeneralRecommender5() {};
};



} // namespace personalized

