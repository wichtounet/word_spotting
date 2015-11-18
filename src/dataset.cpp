//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <fstream>
#include <sstream>

#include <dirent.h>

#include "dataset.hpp"
#include "config.hpp"

namespace {

void read_word_labels(spot_dataset& dataset, const std::string& path) {
    std::ifstream word_labels_stream(path + "/ground_truth/word_labels.txt");

    while (!word_labels_stream.eof()) {
        std::string image_name;
        word_labels_stream >> image_name;

        std::string label;
        word_labels_stream >> label;

        if (image_name.empty() || label.empty()) {
            continue;
        }

        std::istringstream ss(label);
        std::string token;

        while (std::getline(ss, token, '-')) {
            dataset.word_labels[image_name].push_back(token);
        }
    }
}

void read_line_transcriptions(spot_dataset& dataset, const std::string& path) {
    std::ifstream line_transcriptions_stream(path + "/ground_truth/transcription.txt");

    while (!line_transcriptions_stream.eof()) {
        std::string image_name;
        line_transcriptions_stream >> image_name;

        std::string label;
        line_transcriptions_stream >> label;

        if (image_name.empty() || label.empty()) {
            continue;
        }

        std::istringstream ss(label);
        std::string word_token;

        while (std::getline(ss, word_token, '|')) {
            std::vector<std::string> word_labels;

            std::istringstream ss2(word_token);
            std::string token;

            while (std::getline(ss2, token, '-')) {
                word_labels.push_back(token);
            }

            dataset.line_transcriptions[image_name].emplace_back(std::move(word_labels));
        }
    }
}

void read_images(std::unordered_map<std::string, cv::Mat>& map, const std::string& file_path) {
    struct dirent* entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".png") != file_name.size() - 4) {
            continue;
        }

        std::string full_name(file_path + "/" + file_name);

        map[file_name] = cv::imread(full_name, CV_LOAD_IMAGE_ANYDEPTH);

        if (!map[file_name].data) {
            std::cout << "Impossible to read image " << full_name << std::endl;
        }
    }
}

void read_line_images(spot_dataset& dataset, const std::string& path) {
    read_images(dataset.line_images, path + "/data/line_images_normalized/");
}

void read_word_images(spot_dataset& dataset, const std::string& path) {
    read_images(dataset.word_images, path + "/data/word_images_normalized/");
}

void read_list(std::vector<std::string>& list, const std::string& path) {
    std::ifstream stream(path);

    while (!stream.eof()) {
        std::string image_name;
        stream >> image_name;

        if (!image_name.empty()) {
            list.push_back(image_name);
        }
    }
}

void read_keywords(std::vector<std::vector<std::string>>& list, const std::string& path) {
    std::ifstream stream(path);

    while (!stream.eof()) {
        std::string keywords;
        stream >> keywords;

        if (!keywords.empty()) {
            list.emplace_back();

            std::istringstream ss(keywords);
            std::string token;

            while (std::getline(ss, token, '-')) {
                list.back().push_back(token);
            }
        }
    }
}

void read_keywords_iam(std::vector<std::vector<std::string>>& list, const std::string& path) {
    std::ifstream stream(path);

    while (!stream.eof()) {
        std::string line_keywords;
        std::getline(stream, line_keywords);

        std::string keywords(line_keywords.begin(), line_keywords.begin() + line_keywords.find(' '));

        if (!keywords.empty()) {
            list.emplace_back();

            std::istringstream ss(keywords);
            std::string token;

            while (std::getline(ss, token, '-')) {
                list.back().push_back(token);
            }
        }
    }
}

void load_sets_washington(spot_dataset& dataset, const std::string& path) {
    std::string file_path(path + "/sets");

    struct dirent* entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 2) {
            continue;
        }

        std::string full_name(file_path + "/" + file_name);

        read_list(dataset.sets[file_name].test_set, full_name + "/test.txt");
        read_list(dataset.sets[file_name].train_set, full_name + "/train.txt");
        read_list(dataset.sets[file_name].validation_set, full_name + "/valid.txt");
        read_keywords(dataset.sets[file_name].keywords, full_name + "/keywords.txt");
    }
}

void load_sets_parzival(spot_dataset& dataset, const std::string& path) {
    std::string full_name(path + "/sets1/");

    read_list(dataset.sets["cv1"].test_set, full_name + "/test.txt");
    read_list(dataset.sets["cv1"].train_set, full_name + "/train.txt");
    read_list(dataset.sets["cv1"].validation_set, full_name + "/valid.txt");
    read_keywords(dataset.sets["cv1"].keywords, full_name + "/keywords.txt");
}

void load_sets_iam(spot_dataset& dataset, const std::string& path) {
    std::string full_name(path + "/sets/");

    read_list(dataset.sets["cv1"].test_set, full_name + "/test.txt");
    read_list(dataset.sets["cv1"].train_set, full_name + "/train.txt");
    read_list(dataset.sets["cv1"].validation_set, full_name + "/valid.txt");
    read_keywords_iam(dataset.sets["cv1"].keywords, full_name + "/keywords.txt");
}

} //end of anonymous namespace

spot_dataset read_washington(const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images(dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_washington(dataset, path);

    return dataset;
}

spot_dataset read_parzival(const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images(dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_parzival(dataset, path);

    return dataset;
}

spot_dataset read_iam(const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images(dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_iam(dataset, path);

    return dataset;
}

std::vector<std::vector<std::string>> select_keywords(const spot_dataset& dataset, const spot_dataset_set& set, names train_word_names, names test_image_names) {
    std::vector<std::vector<std::string>> keywords;

    for (auto& keyword : set.keywords) {
        bool found = false;

        for (auto& labels : dataset.word_labels) {
            if (keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()) {
                found = true;
                break;
            }
        }

        if (found) {
            auto total_test = std::count_if(test_image_names.begin(), test_image_names.end(),
                                            [&dataset, &keyword](auto& i) { return dataset.word_labels.at({i.begin(), i.end() - 4}) == keyword; });

            if (total_test > 0) {
                keywords.push_back(keyword);
            }
        }
    }

    std::cout << "Selected " << keywords.size() << " keyword out of " << set.keywords.size() << std::endl;

    return keywords;
}
