//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <fstream>
#include <sstream>

#include <dirent.h>

#include "dataset.hpp"
#include "config.hpp"
#include "utils.hpp"

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

void read_word_labels_ak(spot_dataset& dataset, const std::string& path) {
    std::ifstream word_labels_stream(path + "/ground_truth/word_labels.txt");

    std::unordered_map<std::string, std::string> encoded;

    while (!word_labels_stream.eof()) {
        std::string image_name;
        word_labels_stream >> image_name;

        std::string label;
        word_labels_stream >> label;

        if (image_name.empty() || label.empty()) {
            continue;
        }

        // Get rid of AK crap
        if (label == "N/A" || label == "." || label == "," || label == "\"" || label == "-") {
            continue;
        }

        auto unfuck = [](std::string& raw){
            if(raw == std::string("Ä")){
                raw = std::string("ä");
            }

            if(raw == std::string("Ö")){
                raw = std::string("ö");
            }

            if(raw == std::string("Ü")){
                raw = std::string("ü");
            }
        };

        //std::cout << "Before: " << label << std::endl;

        std::transform(label.begin(), label.end(), label.begin(), ::tolower);

        //std::cout << "After: " << label << std::endl;

        for(size_t i = 0; i < label.size(); ++i){
            auto c = label[i];

            if (( c & 0x80 ) == 0 ){
                // The lead bit is zero, this must be ASCII
                dataset.word_labels[image_name].push_back(std::string() + c);
            } else if (( c & 0xE0 ) == 0xC0 ){
                // This indicates encoding on two octets

                std::string raw;
                raw += c;
                raw += label[++i];

                unfuck(raw);

                if(encoded.find(raw) == encoded.end()){
                    encoded[raw] = std::string("E") + std::to_string(encoded.size());
                }

                dataset.word_labels[image_name].push_back(encoded[raw]);
            } else if (( c & 0xF0 ) == 0xE0 ){
                // This indicates encoding on three octets

                std::string raw;
                raw += c;
                raw += label[++i];
                raw += label[++i];

                unfuck(raw);

                if(encoded.find(raw) == encoded.end()){
                    encoded[raw] = std::string("E") + std::to_string(encoded.size());
                }

                dataset.word_labels[image_name].push_back(encoded[raw]);
            } else if (( c & 0xF8 ) == 0xF0 ){
                // This indicates encoding on three octets

                std::string raw;
                raw += c;
                raw += label[++i];
                raw += label[++i];
                raw += label[++i];

                unfuck(raw);

                if(encoded.find(raw) == encoded.end()){
                    encoded[raw] = std::string("E") + std::to_string(encoded.size());
                }

                dataset.word_labels[image_name].push_back(encoded[raw]);
            } else {
                std::cerr << "FUCK CHARACTER " << c << std::endl;
            }
        }

        //std::cout << "Final: " << dataset.word_labels[image_name] << std::endl;
    }

    //for(auto e : encoded){
        //std::cout << e.first << ":" << e.second << std::endl;
    //}

    //std::exit(0);
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
    std::cout << "Read images from '" << file_path << "'" << std::endl;

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

void read_images_ak(const config& conf, std::unordered_map<std::string, cv::Mat>& map, const std::string& file_path, const std::string& sub) {
    std::cout << "Read images from '" << file_path << "/" << sub << "'" << std::endl;

    auto full_path = file_path + "/" + sub;

    struct dirent* entry;
    auto dir = opendir(full_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".png") != file_name.size() - 4) {
            continue;
        }

        std::string full_name(full_path + "/" + file_name);

        auto key = sub + "/" + file_name;

        cv::Mat base_image = cv::imread(full_name, CV_LOAD_IMAGE_ANYDEPTH);

        // Some methods need resizing
        if (conf.method == Method::Patches || conf.method == Method::Terasawa2009 || conf.method == Method::Rodriguez2008) {
            if (base_image.size().height == HEIGHT) {
                map[key] = base_image;
            } else {
                float ratio = float(HEIGHT) / base_image.size().height;

                cv::Mat scaled_normalized(
                    cv::Size(static_cast<size_t>(base_image.size().width * ratio), HEIGHT),
                    CV_8U);
                cv::resize(base_image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
                cv::adaptiveThreshold(scaled_normalized, map[key], 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
            }
        } else {
            map[key] = base_image;
        }

        if (!map[key].data) {
            std::cout << "Impossible to read image " << full_name << std::endl;
        }
    }
}

void read_line_images(spot_dataset& dataset, const std::string& path) {
    read_images(dataset.line_images, path + "/data/line_images_normalized/");
}

void read_word_images(const config& /*conf*/, spot_dataset& dataset, const std::string& path) {
    read_images(dataset.word_images, path + "/data/word_images_normalized/");
}

void read_word_images_gw(const config& conf, spot_dataset& dataset, const std::string& path) {
    if(conf.gray){
        read_images(dataset.word_images, path + "/data/word_gray/");
    } else if(conf.binary){
        read_images(dataset.word_images, path + "/data/word_binary/");
    } else {
        read_images(dataset.word_images, path + "/data/word_images_normalized/");
    }
}

void read_word_images_ak(const config& conf, spot_dataset& dataset, const std::string& path) {
    read_images_ak(conf, dataset.word_images, path + "/data/word_images_normalized", "test");
    read_images_ak(conf, dataset.word_images, path + "/data/word_images_normalized", "train");
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

        if(line_keywords.empty()){
            continue;
        }

        std::string keywords(line_keywords.begin(), line_keywords.begin() + line_keywords.find(' '));

        if(keywords.empty()){
            continue;
        }

        list.emplace_back();

        std::istringstream ss(keywords);
        std::string token;

        while (std::getline(ss, token, '-')) {
            list.back().push_back(token);
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

void load_sets_ak(spot_dataset& dataset, const std::string& path) {
    std::string set_name = "cv1";

    auto& test_set  = dataset.sets[set_name].test_set;
    auto& train_set = dataset.sets[set_name].train_set;
    auto& valid_set = dataset.sets[set_name].validation_set;

    for(auto& image : dataset.word_images){
        if(image.first.substr(0, 5) == "train"){
            train_set.push_back(image.first);
        } else if(image.first.substr(0, 4) == "test"){
            test_set.push_back(image.first);
        }
    }

    valid_set.clear(); // Nothing
}

void load_sets_parzival(spot_dataset& dataset, const std::string& path) {
    std::string full_name(path + "/sets1/");

    read_list(dataset.sets["cv1"].test_set, full_name + "/test.txt");
    read_list(dataset.sets["cv1"].train_set, full_name + "/train.txt");
    read_list(dataset.sets["cv1"].validation_set, full_name + "/valid.txt");
    read_keywords(dataset.sets["cv1"].keywords, full_name + "/keywords.txt");
}

void load_sets_iam(spot_dataset& dataset, const std::string& path) {
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
        read_keywords_iam(dataset.sets[file_name].keywords, full_name + "/keywords.txt");
    }
}

} //end of anonymous namespace

spot_dataset read_washington(const config& conf, const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images_gw(conf, dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_washington(dataset, path);

    return dataset;
}

spot_dataset read_ak(const config& conf, const std::string& path) {
    spot_dataset dataset;

    read_word_labels_ak(dataset, path);
    read_word_images_ak(conf, dataset, path);

    // Now we need to filter the crap out of the data set

    auto it = dataset.word_images.begin();

    while (it != dataset.word_images.end()) {
        if (!dataset.word_labels.count({it->first.begin(), it->first.end() - 4})) {
            it = dataset.word_images.erase(it);
        } else {
            ++it;
        }
    }

    load_sets_ak(dataset, path);

    return dataset;
}

spot_dataset read_botany(const config& conf, const std::string& path) {
    // Seems the same crap
    return read_ak(conf, path);
}

spot_dataset read_manmatha(const config& conf, const std::string& path) {
    spot_dataset dataset;

    read_word_images(conf, dataset, path);

    load_sets_washington(dataset, path);

    return dataset;
}

spot_dataset read_parzival(const config& conf, const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images(conf, dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_parzival(dataset, path);

    return dataset;
}

spot_dataset read_iam(const config& conf, const std::string& path) {
    spot_dataset dataset;

    read_word_labels(dataset, path);
    read_word_images(conf, dataset, path);

    if(dataset_read_lines){
        read_line_transcriptions(dataset, path);
        read_line_images(dataset, path);
    }

    load_sets_iam(dataset, path);

    return dataset;
}

std::vector<std::vector<std::string>> select_keywords(const config& conf, const spot_dataset& dataset, const spot_dataset_set& set, names train_word_names, names test_image_names, bool verbose) {
    std::vector<std::vector<std::string>> keywords;

    if(conf.ak || conf.botany){
        std::set<std::vector<std::string>> base_keywords;

        for (auto& labels : dataset.word_labels) {
            base_keywords.insert(labels.second);
        }

        for (auto& keyword : base_keywords) {
            bool found = false;

            for (auto& labels : dataset.word_labels) {
                if (keyword == labels.second && std::find(train_word_names.begin(), train_word_names.end(), labels.first) != train_word_names.end()) {
                    found = true;
                    break;
                }
            }

            if (found) {
                auto total_test = std::count_if(test_image_names.begin(), test_image_names.end(),
                                                [&dataset, &keyword](auto& i) {
                                                    std::string key{i.begin(), i.end() - 4};
                                                    return dataset.word_labels.at(key) == keyword;
                                                });

                if (total_test > 0) {
                    keywords.push_back(keyword);
                }
            }
        }
    } else {
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
    }

    if(verbose){
         std::cout << "Selected " << keywords.size() << " keyword out of " << set.keywords.size() << std::endl;
    }

    return keywords;
}
