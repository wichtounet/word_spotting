//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <fstream>
#include <sstream>

#include <dirent.h>

#include "washington.hpp"

namespace {

void read_word_labels(washington_dataset& dataset, const std::string& path){
    std::ifstream word_labels_stream(path + "/ground_truth/word_labels.txt");

    while(!word_labels_stream.eof()){
        std::string image_name;
        word_labels_stream >> image_name;

        std::string label;
        word_labels_stream >> label;

        if(image_name.empty() || label.empty()){
            continue;
        }

        std::istringstream ss(label);
        std::string token;

        while(std::getline(ss, token, '-')) {
            dataset.word_labels[image_name].push_back(token);
        }
    }
}

void read_line_transcriptions(washington_dataset& dataset, const std::string& path){
    std::ifstream line_transcriptions_stream(path + "/ground_truth/transcription.txt");

    while(!line_transcriptions_stream.eof()){
        std::string image_name;
        line_transcriptions_stream >> image_name;

        std::string label;
        line_transcriptions_stream >> label;

        if(image_name.empty() || label.empty()){
            continue;
        }

        std::istringstream ss(label);
        std::string word_token;

        while(std::getline(ss, word_token, '|')) {
            std::vector<std::string> word_labels;

            std::istringstream ss2(word_token);
            std::string token;

            while(std::getline(ss2, token, '-')) {
                word_labels.push_back(token);
            }

            dataset.line_transcriptions[image_name].emplace_back(std::move(word_labels));
        }
    }
}

void read_images(std::unordered_map<std::string, cv::Mat>& map, const std::string& file_path){
    struct dirent *entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if(file_name.size() <= 3 || file_name.find(".png") == std::string::npos){
            continue;
        }

        std::string full_name(file_path + "/" + file_name);

        map[file_name] = cv::imread(full_name.c_str(), 1);

        if(!map[file_name].data){
            std::cout << "Impossible to read image " << full_name << std::endl;
        }
    }
}

void read_line_images(washington_dataset& dataset, const std::string& path){
    read_images(dataset.line_images, path + "/data/line_images_normalized/");
}

void read_word_images(washington_dataset& dataset, const std::string& path){
    read_images(dataset.word_images, path + "/data/word_images_normalized/");
}

void read_list(std::vector<std::string>& list, const std::string& path){
    std::ifstream stream(path);

    while(!stream.eof()){
        std::string image_name;
        stream >> image_name;

        if(!image_name.empty()){
            list.push_back(image_name);
        }
    }
}

void read_keywords(std::vector<std::vector<std::string>>& list, const std::string& path){
    std::ifstream stream(path);

    while(!stream.eof()){
        std::string keywords;
        stream >> keywords;

        if(!keywords.empty()){
            list.emplace_back();

            std::istringstream ss(keywords);
            std::string token;

            while(std::getline(ss, token, '-')) {
                list.back().push_back(token);
            }
        }
    }
}

void load_sets(washington_dataset& dataset, const std::string& path){
    std::string file_path(path + "/sets");

    struct dirent *entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if(file_name.size() <= 2){
            continue;
        }

        std::string full_name(file_path + "/" + file_name);

        read_list(dataset.sets[file_name].test_set, full_name + "/test.txt");
        read_list(dataset.sets[file_name].train_set, full_name + "/train.txt");
        read_list(dataset.sets[file_name].validation_set, full_name + "/valid.txt");
        read_keywords(dataset.sets[file_name].keywords, full_name + "/keywords.txt");
    }
}

} //end of anonymous namespace

washington_dataset read_dataset(const std::string& path){
    washington_dataset dataset;

    read_word_labels(dataset, path);
    read_line_transcriptions(dataset, path);

    read_line_images(dataset, path);
    read_word_images(dataset, path);

    load_sets(dataset, path);

    return dataset;
}
