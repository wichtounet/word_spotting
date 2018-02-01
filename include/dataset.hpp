//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_HPP
#define WORD_SPOTTER_HPP

#include <unordered_map>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "config.hpp"

using names = const std::vector<std::string>&;

struct spot_dataset_set {
    std::vector<std::vector<std::string>> keywords;
    std::vector<std::string> test_set;
    std::vector<std::string> train_set;
    std::vector<std::string> validation_set;
};

struct spot_dataset {
    std::unordered_map<std::string, std::vector<std::string>> word_labels;
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> line_transcriptions;

    std::unordered_map<std::string, cv::Mat> line_images;
    std::unordered_map<std::string, cv::Mat> word_images;

    std::unordered_map<std::string, spot_dataset_set> sets;
};

spot_dataset read_washington(const config& conf, const std::string& path);
spot_dataset read_manmatha(const config& conf, const std::string& path);
spot_dataset read_parzival(const config& conf, const std::string& path);
spot_dataset read_iam(const config& conf, const std::string& path);
spot_dataset read_ak(const config& conf, const std::string& path);

std::vector<std::vector<std::string>> select_keywords(const config& conf, const spot_dataset& dataset, const spot_dataset_set& set, names train_word_names, names test_image_names, bool verbose = true);

#endif
