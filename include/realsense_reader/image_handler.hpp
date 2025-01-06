// Copyright 2025 Hyeongjun Jeon

#ifndef REALSENSE_READER__IMAGE_HANDLER_HPP_
#define REALSENSE_READER__IMAGE_HANDLER_HPP_

#include <atomic>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>
#include "opencv2/aruco.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"


namespace realsense_reader
{
class ImageHandler
{
public:
  explicit ImageHandler(const int target_camera_index = 0);
  virtual ~ImageHandler() {}
  void run();

private:
  void initialize_stream();
  bool estimate_depth(const cv::Mat & depth_image, cv::Mat & color_image);

  std::string serial_number_;

  rs2::frameset frame_set_;

  rs2::frame color_frame_;
  rs2::frame depth_frame_;

  std::shared_ptr<rs2::pipeline> pipeline_;
  std::shared_ptr<rs2::align> align_to_color_;

  rs2::config config_;

  cv::Mat color_image_;
  cv::Mat depth_image_;
  cv::Point2f depth_range_ = cv::Point2f(0.5f, 10.0f);

  cv::Size2i image_size_ = cv::Size2i(640, 360);
  uint8_t fps_ = 30;
  cv::Mat color_intrinsics_matrix_;
  cv::Mat distortion_matrix_;
};
}  // namespace realsense_reader
#endif  // REALSENSE_READER__IMAGE_HANDLER_HPP_
