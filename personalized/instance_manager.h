// Copyright 2015 Letv Inc. All Rights Reserved.
// Author: sunhaochuan@letv.com (Sun Haochuan)

#pragma once
#include <map>
#include "base/mutex.h"
#include "shared/personalized/logistic_regression.h"
#include "util/gtl/stl_util-inl.h"

namespace personalized{

class InstanceManager {
public:
  Classifier* GetInstance(const std::string& instance_name);
  //the same named instance only can be registered once, first come first served
  bool RegisterInstace(const std::string& instance_name,
                       const std::string& instance_type,
                       const std::map<std::string, std::string>& instance_parameter);
  ~InstanceManager() {
    gtl::STLDeleteValues(&instance_repository_);
  }

private:
  std::map<std::string, Classifier*> instance_repository_;
  base::RwMutex repository_rwmutex_;

  InstanceManager();
  friend struct DefaultSingletonTraits<InstanceManager>;
  DISALLOW_COPY_AND_ASSIGN(InstanceManager);
};

} // namespace personalized

