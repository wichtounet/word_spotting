//=======================================================================
// Copyright Baptiste Wicht 2015-2016.
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

cv::Mat elastic_distort(const cv::Mat& clean_image);

std::pair<int, std::string> exec_command(const std::string& command);

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    std::string comma = "";
    stream << "[";
    for (auto& v : vec) {
        stream << comma << v;
        comma = ", ";
    }
    stream << "]";

    return stream;
}

template <typename T>
std::string keyword_to_string(const std::vector<T>& vec) {
    std::string comma = "";
    std::string result;
    result += "[";
    for (auto& v : vec) {
        result += comma;
        result += v;
        comma = ", ";
    }
    result += "]";

    return result;
}

template <typename T>
std::string keyword_to_short_string(const std::vector<T>& vec) {
    std::string result;
    for (auto& v : vec) {
        result += v;
    }
    return result;
}

etl::dyn_matrix<weight, 3> mat_for_patches(const config& conf, const cv::Mat& image);

template <typename DBN>
typename DBN::template layer_type<0>::input_one_t holistic_mat(const config& conf, const cv::Mat& image) {
    using image_t = typename DBN::template layer_type<0>::input_one_t;

    image_t training_image;

#ifndef OPENCV_23
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    cv::Mat scaled_normalized(cv::Size(std::max(1UL, WIDTH / conf.downscale), std::max(1UL, HEIGHT / conf.downscale)), CV_8U);
    cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
    cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    for (std::size_t y = 0; y < static_cast<std::size_t>(normalized.size().height); ++y) {
        for (std::size_t x = 0; x < static_cast<std::size_t>(normalized.size().width); ++x) {
            auto pixel = normalized.at<uint8_t>(cv::Point(x, y));

            training_image(0, y, x) = pixel == 0 ? 0.0 : 1.0;

            if (pixel != 0 && pixel != 255) {
                std::cout << "The normalized input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }
#else
    cpp_unused(conf);
    cpp_unused(image);
#endif

    return training_image;
}

template <typename DBN>
std::vector<typename DBN::template layer_type<0>::input_one_t> mat_to_patches(const config& conf, const cv::Mat& image, bool train) {
    using image_t = typename DBN::template layer_type<0>::input_one_t;

    cv::Mat buffer_image;

    if (conf.downscale > 1) {
        cv::Mat scaled_normalized(
            cv::Size(std::max(1UL, static_cast<size_t>(image.size().width)), std::max(1UL, image.size().height / conf.downscale)),
            CV_8U);
        cv::resize(image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, buffer_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    const cv::Mat& clean_image = conf.downscale > 1 ? buffer_image : image;

    const auto patch_width        = conf.patch_width;
    const auto left               = patch_width / 2;
    const auto patch_stride       = train ? conf.train_stride : conf.test_stride;
    const std::size_t real_width  = clean_image.size().width;
    const std::size_t real_height = clean_image.size().height;

    cv::Mat final_image(cv::Size(real_width + patch_width, real_height), CV_8U);
    final_image = cv::Scalar(1.0);

#ifdef OPENCV_23
    //This is not efficient, but we need this because of the ANCIENT retarded grid machines
    for (std::size_t y = 0; y < real_height; ++y) {
        for (std::size_t x = 0; x < real_width; ++x) {
            final_image.at<uint8_t>(y, x + left) = clean_image.at<uint8_t>(y, x);
        }
    }
#else
    clean_image.copyTo(final_image(cv::Rect(left , 0, real_width, real_height)));
#endif

    std::vector<image_t> patches;

    for (std::size_t real_x = 0; real_x < real_width; real_x += patch_stride) {
        patches.emplace_back();

        auto& patch = patches.back();

        for (std::size_t real_y = 0; real_y < real_height; ++real_y) {
            for (std::size_t x = 0; x < patch_width; ++x) {
                patch(0, real_y, x) = final_image.at<uint8_t>(real_y, real_x + x) == 0 ? 0.0 : 1.0;
            }
        }
    }

    return patches;
}

#endif
