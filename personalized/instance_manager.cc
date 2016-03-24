// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#include "shared/personalized/instance_manager.h"
#include "base/letv.h"
#include "base/logging.h"


using namespace std;

namespace personalized  {

InstanceManager::InstanceManager() {
}

Classifier* InstanceManager::GetInstance(const std::string& instance_name) {
  ReaderMutexLock reader_lock(&repository_rwmutex_);
  std::map<std::string, Classifier*>::iterator iter = instance_repository_.find(instance_name);

  return iter == instance_repository_.end() ? NULL : iter->second;
}

bool InstanceManager::RegisterInstace(const std::string& instance_name,
                const std::string& instance_type,
                const std::map<std::string, std::string>& instance_parameter) {
  WriterMutexLock lock(&repository_rwmutex_);

  std::map<std::string, Classifier*>::iterator iter = instance_repository_.find(instance_name);
  if (iter != instance_repository_.end()) {
    LOG(INFO) << "the intance has exist:" << instance_name;
    return false;
  }

  if (instance_type == "LR") {
    Classifier* instance_p = new LogisticRegression();
    CHECK(instance_p->Init(instance_parameter) == true);
    instance_repository_[instance_name] = instance_p;
    return true;
  } else {
    CHECK(false);
    return false;
  }
}


} //namespace personalized
