//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_WASHINGTON_HPP
#define WORD_SPOTTER_WASHINGTON_HPP

#include <unordered_map>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

struct washington_dataset_set {
    std::vector<std::vector<std::string>> keywords;
    std::vector<std::string> test_set;
    std::vector<std::string> train_set;
    std::vector<std::string> validation_set;
};

struct washington_dataset {
    std::unordered_map<std::string, std::vector<std::string>> word_labels;
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> line_transcriptions;

    std::unordered_map<std::string, cv::Mat> line_images;
    std::unordered_map<std::string, cv::Mat> word_images;

    std::unordered_map<std::string, washington_dataset_set> sets;
};

washington_dataset read_dataset(const std::string& path);

#endif
