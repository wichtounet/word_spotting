//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>

#include "utils.hpp"

etl::dyn_matrix<weight> mat_to_dyn(const config& conf, const cv::Mat& image){
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    cv::Mat scaled_normalized(cv::Size(WIDTH / conf.downscale, HEIGHT / conf.downscale), CV_8U);
    cv::resize(normalized, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
    cv::adaptiveThreshold(scaled_normalized, normalized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);

    etl::dyn_matrix<weight> training_image(normalized.size().height, normalized.size().width);

    for(std::size_t y = 0; y < static_cast<std::size_t>(normalized.size().height); ++y){
        for(std::size_t x = 0; x < static_cast<std::size_t>(normalized.size().width); ++x){
            auto pixel = normalized.at<uint8_t>(cv::Point(x, y));

            training_image(y, x) = pixel == 0 ? 0.0 : 1.0;

            if(pixel != 0 && pixel != 255){
                std::cout << "The normalized input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}

etl::dyn_matrix<weight, 3> mat_for_patches(const config& /*conf*/, const cv::Mat& image){
    etl::dyn_matrix<weight, 3> training_image(std::size_t(1), std::size_t(image.size().height), std::size_t(image.size().width));

    for(std::size_t y = 0; y < static_cast<std::size_t>(image.size().height); ++y){
        for(std::size_t x = 0; x < static_cast<std::size_t>(image.size().width); ++x){
            auto pixel = image.at<uint8_t>(cv::Point(x, y));

            training_image(0, y, x) = pixel == 0 ? 0.0 : 1.0;

            if(pixel != 0 && pixel != 255){
                std::cout << "The input image is not binary! pixel:" << static_cast<int>(pixel) << std::endl;
            }
        }
    }

    return training_image;
}

std::vector<etl::dyn_matrix<weight, 3>> mat_to_patches(const config& conf, const cv::Mat& image, bool train){
    cv::Mat buffer_image;

    if(conf.downscale > 1){
        cv::Mat scaled_normalized(cv::Size(image.size().width / conf.downscale, image.size().height / conf.downscale), CV_8U);
        cv::resize(image, scaled_normalized, scaled_normalized.size(), cv::INTER_AREA);
        cv::adaptiveThreshold(scaled_normalized, buffer_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 2);
    }

    const cv::Mat& clean_image = conf.downscale > 1 ? buffer_image : image;

    std::vector<etl::dyn_matrix<weight, 3>> patches;

    const auto context = conf.patch_width / 2;
    const auto patch_stride = train ? conf.train_stride : conf.test_stride;

    for(std::size_t i = 0; i < static_cast<std::size_t>(clean_image.size().width); i += patch_stride){
        patches.emplace_back(
            static_cast<std::size_t>(1),
            static_cast<std::size_t>(clean_image.size().height),
            static_cast<std::size_t>(conf.patch_width));

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
