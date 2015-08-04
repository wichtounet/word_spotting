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
#ifdef OPENCV_23
    etl::dyn_matrix<weight> training_image(HEIGHT, WIDTH);
#else
    cv::Mat normalized(cv::Size(WIDTH, HEIGHT), CV_8U);
    normalized = cv::Scalar(255);

    image.copyTo(normalized(cv::Rect((WIDTH - image.size().width) / 2, 0, image.size().width, HEIGHT)));

    cv::Mat scaled_normalized(cv::Size(std::max(1UL, WIDTH / conf.downscale), std::max(1UL, HEIGHT / conf.downscale)), CV_8U);
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
#endif

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
