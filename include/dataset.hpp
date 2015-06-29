//=======================================================================
// Copyright Baptiste Wicht 2015.
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

spot_dataset read_washington(const std::string& path);
spot_dataset read_parzival(const std::string& path);

using names = const std::vector<std::string>&;

template<typename Dataset, typename Set>
std::vector<std::vector<std::string>> select_keywords(const Dataset& dataset, const Set& set, names train_word_names, names test_image_names){
    std::vector<std::vector<std::string>> keywords;

    for(std::size_t k = 0; k < set.keywords.size(); ++k){
        auto& keyword = set.keywords[k];

        bool found = false;

        for(auto& labels : dataset.word_labels){
            if(keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()){
                found = true;
                break;
            }
        }

        if(found){
            auto total_test = std::count_if(test_image_names.begin(), test_image_names.end(),
                [&dataset, &keyword](auto& i){ return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

            if(total_test > 0){
                keywords.push_back(keyword);
            }
        }
    }

    std::cout << "Selected " << keywords.size() << " keyword out of " << set.keywords.size() << std::endl;

    return keywords;
}

#endif
