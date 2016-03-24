# Copyright 2015 letv Inc. All Rights Reserved.
# Author: sunhaochuan@letv.com

cc_library(
        name = 'personalized',
        srcs = [
                'arena.cc',
                'ctr_predict_feature_builder.cc',
                'general_recommender_feature_builder.cc',
                'happy_to_see_recommender_feature_builder.cc',
                'hash_wrapper.cc',
                'logistic_regression.cc',
                'sized_string.cc',
                'instance_manager.cc',
         		 'general_recommender_with_treelike_user_profile.cc',
         		 'sarrs_extra_feature_builder.cc',
         		 'general_recommender_with_treelike_user_profile_pro.cc'
        ],

        deps = [
                '//base:base',
                '//recommendation/proto:recommendation_param_proto',
                '//recommendation/thrift:database_thrift',
                '//recommendation/thrift:recommendation_thrift',
                '//recommendation/thrift:user_feature_thrift',
                '//util/file:simple_line_reader',
                '//util/registerer:registerer',
                '//shared/serving/serving_context:backend_serving_context',
                '//third_party/libmemcached-0.25:memcached',
        ],
        link_all_symbols=True,
        #optimize = ['g', 'msse4.2', 'O0'],
)

cc_binary(
	name = 'general_recommender_feature_builder',
	srcs = [
			'general_recommender_feature_builder_main.cc',
	],
	deps = [
			':personalized',
	],
)

cc_binary(
	name = 'happy_to_see_recommender_feature_builder',
	srcs = [
			'happy_to_see_recommender_feature_builder_main.cc',
	],
	deps = [
			':personalized',
	],
	#optimize = ['g', 'msse4.2', 'O0'],
)