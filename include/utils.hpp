//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_UTILS_HPP
#define WORD_SPOTTER_UTILS_HPP

#include <vector>
#include <string>

#include "etl/etl.hpp"

#include "config.hpp"
#include "dataset.hpp"

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec){
    std::string comma = "";
    stream << "[";
    for(auto& v : vec){
        stream << comma << v;
        comma = ", ";
    }
    stream << "]";

    return stream;
}

template<typename T>
std::string keyword_to_string(const std::vector<T>& vec){
    std::string comma = "";
    std::string result;
    result += "[";
    for(auto& v : vec){
        result += comma;
        result += v;
        comma = ", ";
    }
    result += "]";

    return result;
}

etl::dyn_matrix<weight> mat_to_dyn(const config& conf, const cv::Mat& image);
etl::dyn_matrix<weight, 3> mat_for_patches(const config& conf, const cv::Mat& image);

template<typename DBN>
std::vector<typename DBN::template layer_type<0>::input_one_t> mat_to_patches(const config& conf, const cv::Mat& image, bool train){
    using image_t = typename DBN::template layer_type<0>::input_one_t;

    cv::Mat buffer_image;

    if(conf.downscale > 1){
        cv::Mat scaled_normalized(
            cv::Size(std::max(1UL, image.size().width / conf.downscale), std::max(1UL, image.size().height / conf.downscale)),
            CV_8U);
        cv::resize(image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, buffer_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    const cv::Mat& clean_image = conf.downscale > 1 ? buffer_image : image;

    std::vector<image_t> patches;

    const auto context = conf.patch_width / 2;
    const auto patch_stride = train ? conf.train_stride : conf.test_stride;

    for(std::size_t i = 0; i < static_cast<std::size_t>(clean_image.size().width); i += patch_stride){
        patches.emplace_back();

        auto& patch = patches.back();

        for(std::size_t y = 0; y < static_cast<std::size_t>(clean_image.size().height); ++y){
            for(int x = i - context; x < static_cast<int>(i + context); ++x){
                uint8_t pixel = 1;

                if(x >= 0 && x < clean_image.size().width){
                    //pixel = image.at<uint8_t>(y, x + i * patch_stride);
                    pixel = clean_image.at<uint8_t>(y, x);
                }

                patch(0, y, x - i + context) = pixel == 0 ? 0.0 : 1.0;
            }
        }
    }

    return patches;
}

#endif
