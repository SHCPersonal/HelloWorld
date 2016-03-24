// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "happy_to_see_recommender_feature_builder.h"
#include "base/thrift.h"
#include "base/string_util.h"
#include "base/logging.h"
#include "base/flags.h"
#include "base/hash.h"
#include "util/global_init/global_init.h"
#include "util/encode/base64encoder.h"
#include <string>
#include <iostream>

using namespace std;
using namespace personalized;

int main(int argc, char** argv) {

  string uid(argv[1]);
  string user_profile_string(argv[2]);
  string rec_vid(argv[3]);
  string media_doc_info_string(argv[4]);
  string ctr_string(argv[5]);
  string time_string(argv[6]);

  if (ctr_string == "test") {
    ctr_string = "";
  }

  //HappyToSeeRecommenderFeatureBuilder::PythonUtil _builder;
  string res;
  for (int i = 0; i < ctr_string.size(); ++i) {
    if (ctr_string[i] == '?') {
      ctr_string[i] = ' ';
    }
  }

  //sleep(20);

  /*res = _builder.BuildFeature(
                uid,
                user_profile_string,
                rec_vid,
                media_doc_info_string,
                ctr_string,
                time_string);*/


  printf ("%s", res.c_str());
}

