// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "logistic_regression.h"
#include "base/letv.h"
#include "base/logging.h"

using namespace std;

namespace personalized {

// the class LogisticRegression
LogisticRegression::LogisticRegression()
  : init_bool_(false),
    ctr_data_(NULL),
    model_data_(NULL),
    file_monitor_p_(NULL) {}

float LogisticRegression::Predict(
    const vector<SizedStringBuilder >& feature_list) {
  ReaderMutexLock reader_lock(&model_rwmutex_);
  if (!init_bool_) {
    return -1.0;
  }

  float res_f = 0;
  float linear_sum = 0;

  vector<SizedStringBuilder>::const_iterator iter;
  for (iter = feature_list.begin(); iter != feature_list.end(); ++iter) {
    IterModelData iter_feature = model_data_->find(*iter);
    if (iter_feature != model_data_->end()) {
      if (iter->weight== 1.0f) {
        linear_sum += iter_feature->second;
      } else {
        linear_sum += iter_feature->second * iter->weight;
      }
    }
  }

  res_f = 1.0f / (1.0f + exp(0.0f - linear_sum));
  VLOG(1) << "CTR Predict detail, linear_sum: " << linear_sum << " res_f: " << res_f;
  return res_f;
}

void LogisticRegression::Debug(
  const vector<SizedStringBuilder >& feature_list,
  vector<string>& feature_res) {
  ReaderMutexLock reader_lock(&model_rwmutex_);
  if (!init_bool_) {
    CHECK(false);
  }

  feature_res.clear();
  vector<SizedStringBuilder>::const_iterator iter;
  for (iter = feature_list.begin(); iter != feature_list.end(); ++iter) {
    IterModelData  iter_feature = model_data_->find(*iter);
    if (iter_feature != model_data_->end()) {
      std::ostringstream res_str;
      res_str << iter->str() << ":" << iter->weight << ":" << iter_feature->second;
      feature_res.push_back(res_str.str());
    }
  }

}

bool LogisticRegression::LoadModel(
    const std::string& model_file, ModelData* model_data, FeatureData*
                 user_feature, FeatureData* item_feature, Arena* arena) {

  ifstream f_in(model_file.c_str(), ifstream::in);
  if (!f_in) {
    init_bool_=false;
    return false;
  }
  string line;
  int cnt = 0;
  SizedString tmp_str;
  while (getline(f_in, line)) {
    cnt++;
    SizedStringBuilder tmp_str_builder;
    if(!tmp_str_builder.load(line, arena)) {
      LOG(INFO) << "bad ctr model line: " << line << endl;
      continue;
    }
    SizedStringBuilder tmp_builder;
    (*model_data)[tmp_str_builder] = tmp_str_builder.weight;
    /*if (*(tmp_str_builder.builder[0].pcData) > '9')
    {
      continue;
    }
    for (int i = 1; i < tmp_str_builder.size; i++) {
      char c = *(tmp_str_builder.builder[0].pcData + (i * 2) - 1);
      if (c == '1') {
        user_feature->insert(tmp_str_builder.builder[i]);
      } else if (c == '2') {
        item_feature->insert(tmp_str_builder.builder[i]);
      }
    }*/
  }

  LOG(INFO) << "load the CTR model from:" << model_file << " with line:" << cnt << endl;
  LOG(INFO) << "model contain: " << model_data->size() << " keys" <<endl;
  LOG(INFO) << "user feature contain: " << user_feature->size() << " keys" << endl;
  LOG(INFO) << "item feature contain: " << item_feature->size() << " keys" << endl;
  init_bool_=true;
  return true;
}

bool LogisticRegression::GetCTRFeature(
        const std::string& vid,
        CTR_feature& ctr_feature) {
  ReaderMutexLock lock(&ctr_data_rwmutex_);

  CtrData::iterator it = ctr_data_->find(vid);
  if (it != ctr_data_->end()) {
    ctr_feature = it->second;
    return true;
  }
  return false;
}

bool LogisticRegression::IsContainUser(SizedString& feature) {
  ReaderMutexLock lock(&model_rwmutex_);

  if (user_feature_set_->find(feature) != user_feature_set_->end()) {
    return true;
  }
  return false;
}

bool LogisticRegression::IsContainItem(SizedString& feature) {
  ReaderMutexLock lock(&model_rwmutex_);

  if (item_feature_set_->find(feature) != item_feature_set_->end()) {
    return true;
  }
  return false;
}

std::string LogisticRegression::GetFeatureBuilderType() {
    return feature_builder_type_;
}

void LogisticRegression::RefreshModel() {
  CHECK(model_type_string_ == "logistic_regression");
  scoped_ptr<ModelData> tmp_model(new ModelData);
  scoped_ptr<FeatureData> tmp_user_feature(new FeatureData);
  scoped_ptr<FeatureData> tmp_item_feature(new FeatureData);
  scoped_ptr<Arena> tmp_arena(new Arena);
  tmp_model->set_load_factor(0.5);
  tmp_user_feature->max_load_factor(0.5);
  tmp_item_feature->max_load_factor(0.5);
  if (!LoadModel(model_file_string_, tmp_model.get(), tmp_user_feature.get(),
                  tmp_item_feature.get(), tmp_arena.get())) {
    LOG(ERROR) << "CTR preidct faild to load the ctr model";
  }
  {
    WriterMutexLock lock(&model_rwmutex_);
    model_data_.swap(tmp_model);
    user_feature_set_.swap(tmp_user_feature);
    item_feature_set_.swap(tmp_item_feature);
    model_memory_.swap(tmp_arena);
  }
}

void LogisticRegression::RefreshCTR() {
  scoped_ptr<CtrData> tmp_ctr_data(new CtrData);
  tmp_ctr_data->set_empty_key("");
  if (!LoadCTR(tmp_ctr_data.get())) {
    LOG(ERROR) << "CTR predict failed to load the ctr file" ;
  }
  {
    WriterMutexLock lock(&ctr_data_rwmutex_);
    ctr_data_.swap(tmp_ctr_data);
  }
}


bool LogisticRegression::Init(const std::map<string, string>& parameter) {
  MutexLock lock(&init_mutex_);
  if (init_bool_) {
    return true;
  }
  CHECK(!init_bool_);
  CHECK(parameter.count("model_type") == 1);
  CHECK(parameter.count("model_file") == 1);
  CHECK(parameter.count("ctr_file") == 1);
  CHECK(parameter.count("feature_builder_type") == 1);

  model_type_string_ = parameter.find("model_type")->second;
  model_file_string_ = parameter.find("model_file")->second;
  ctr_data_file_string_ = parameter.find("ctr_file")->second;
  feature_builder_type_ = parameter.find("feature_builder_type")->second;    

  n_ctr_rate_string_ = "n_recvid_rate";
  y_ctr_rate_string_ = "y_recvid_rate";

  CHECK(!feature_builder_type_.empty());
  // init the model
  CHECK(model_type_string_ == "logistic_regression");

  RefreshModel();
  RefreshCTR();

  // start the monitor
  file_monitor_p_ = util::FileMonitor::GetInstance();
  base::Closure* callback =
      base::NewPermanentCallback(this, &LogisticRegression::RefreshModel);
  file_monitor_p_->Register(model_file_string_, callback);

  base::Closure* callback_ctr =
      base::NewPermanentCallback(this, &LogisticRegression::RefreshCTR);
  file_monitor_p_->Register(ctr_data_file_string_, callback_ctr);
  init_bool_ = true;
  return true;
}

bool LogisticRegression::LoadCTR(CtrData* ctr_data) {
  ifstream f_in(ctr_data_file_string_.c_str(), ifstream::in);
  if (!f_in) return false;

  string line;
  int cnt = 0;
  while (getline(f_in, line)) {
    vector<string> items;
    cnt++;

    string vid;
    CTR_feature ctr;
    if (ctr.Load(line, vid)) {
      if (!vid.empty()) {
        (*ctr_data)[vid] = ctr;
      }
    } else {
      LOG(ERROR) << "CTR preidct BAD CTR FEATURE";
    }
  }
  LOG(INFO) << "load the click_data from:" << ctr_data_file_string_ << " with line:" << cnt;
  return !ctr_data->empty();
}

} //namespace personalized


