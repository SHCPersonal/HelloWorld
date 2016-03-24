// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "general_recommender_feature_builder.h"
#include "general_recommender_with_treelike_user_profile.h"
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

/*int main(int argc, char** argv) {

  string uid(argv[1]);
  string user_profile_string(argv[2]);
  string rec_vid(argv[3]);
  string media_doc_info_string(argv[4]);
  string ctr_string(argv[5]);
  string time_string(argv[6]);

  if (ctr_string == "test") {
    ctr_string = "";
  }
  cout << "start" << endl;
  Arena arena;
  CTRPredictFeatureBuilder* _builder = new GeneralRecommender1(true, &arena);;
  string res;
  for (int i = 0; i < ctr_string.size(); ++i) {
    if (ctr_string[i] == '?') {
      ctr_string[i] = ' ';
    }
  }
  sleep(20);
  vector<string> history;
    res = _builder->BuildFeatureOffline(
                  uid,
                  user_profile_string,
                  rec_vid,
                  media_doc_info_string,
                  "",
                  ctr_string,
                  time_string,
                  history);

  printf ("%s", res.c_str());
}*/

int main(int argc, char** argv) {
  string line;
  vector<string> history;
  while(std::getline(std::cin, line)) {
    vector<string> items;
    SplitString(line, '\t', &items);
    string uid = items[0];
    string user_profile_string = items[1];
    string rec_vid= items[2];
    string media_doc_info_string= items[3];
    string media_doc_info_pid_string= items[4];
    string ctr_string= items[5];
    string time_string= items[6];
    Arena arena;
    CTRPredictFeatureBuilder* _builder = new GeneralRecommender1(true, &arena);;
    string res;
    for (int i = 0; i < ctr_string.size(); ++i) {
      if (ctr_string[i] == '?') {
        ctr_string[i] = ' ';
      }
    }
    //sleep(20);
    vector<string> history;
    res = _builder->BuildFeatureOffline(
                  uid,
                  user_profile_string,
                  rec_vid,
                  media_doc_info_string,
                  media_doc_info_pid_string,
                  ctr_string,
                  time_string,
                  history);

    printf ("%s", res.c_str());
  }
}


