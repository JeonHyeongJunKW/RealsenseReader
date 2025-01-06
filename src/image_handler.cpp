// Copyright 2025 Hyeongjun Jeon

#include "realsense_reader/image_handler.hpp"


realsense_reader::ImageHandler::ImageHandler(const int target_camera_index)
{
  rs2::context context;
  rs2::device_list device_list = context.query_devices();
  if (device_list.size() == 0) {
    throw std::runtime_error("No device detected. Is it plugged in?");
  }

  if (device_list.size() <= target_camera_index) {
    const std::string error_log =
      "Invalid target camera index (device count: " +
      std::to_string(device_list.size()) +
      ")";
    throw std::runtime_error(error_log);
  }
  auto target_device = device_list[target_camera_index];
  serial_number_ = target_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
  for (auto & sensor : target_device.query_sensors()) {
    const std::string sensor_name = sensor.get_info(RS2_CAMERA_INFO_NAME);
    if (sensor_name == "Stereo Module") {
      sensor.set_option(RS2_OPTION_LASER_POWER, 360.0);
      sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, true);
      sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, 0);

      rs2::region_of_interest roi;
      roi.min_x = 80;
      roi.max_x = 560;
      roi.min_y = 45;
      roi.max_y = 180;

      sensor.as<rs2::roi_sensor>().set_region_of_interest(roi);
    } else if (sensor_name == "RGB Camera") {
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, true);
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, false);
    }
  }

  config_.enable_device(serial_number_);
  config_.enable_stream(
    RS2_STREAM_COLOR,
    image_size_.width,
    image_size_.height,
    RS2_FORMAT_BGR8,
    fps_);
  config_.enable_stream(
    RS2_STREAM_DEPTH,
    image_size_.width,
    image_size_.height,
    RS2_FORMAT_Z16,
    fps_);

  align_to_color_ = std::make_shared<rs2::align>(RS2_STREAM_COLOR);
  pipeline_ = std::make_shared<rs2::pipeline>();
  this->initialize_stream();
}

void realsense_reader::ImageHandler::initialize_stream()
{
  rs2::pipeline_profile pipeline_profile = pipeline_->start(config_);

  auto color_stream = pipeline_profile.get_stream(RS2_STREAM_COLOR);
  auto color_intrinsics = color_stream.as<rs2::video_stream_profile>().get_intrinsics();

  color_intrinsics_matrix_ = cv::Mat::eye(3, 3, CV_32F);
  color_intrinsics_matrix_.at<double>(0, 0) = color_intrinsics.fx;
  color_intrinsics_matrix_.at<double>(0, 2) = color_intrinsics.ppx;
  color_intrinsics_matrix_.at<double>(1, 1) = color_intrinsics.fy;
  color_intrinsics_matrix_.at<double>(1, 2) = color_intrinsics.ppy;
  color_intrinsics_matrix_.at<double>(2, 2) = 1.0f;

  pipeline_->stop();
}

void realsense_reader::ImageHandler::run()
{
  pipeline_->start(config_);
  std::cout << "Started to capture images" << std::endl;
  std::cout << "Press ESC key to stop" << std::endl;

  while (cv::waitKey(1) != 27) {
    frame_set_ = pipeline_->wait_for_frames();
    frame_set_ = align_to_color_->process(frame_set_);  // align only for depth

    color_frame_ = frame_set_.get_color_frame();
    depth_frame_ = frame_set_.get_depth_frame();

    color_image_ = cv::Mat(
      image_size_.height,
      image_size_.width,
      CV_8UC3,
      const_cast<void *>(color_frame_.get_data()));

    depth_image_ = cv::Mat(
      image_size_.height,
      image_size_.width,
      CV_16UC1,
      const_cast<void *>(depth_frame_.get_data()));

    cv::Mat output_confidence;
    depth_image_.convertTo(depth_image_, CV_32FC1, 0.001);

    cv::threshold(depth_image_, depth_image_, 10.0f, 0.0f, cv::THRESH_TRUNC);
    depth_image_ = depth_image_ * 25.5f;
    depth_image_.convertTo(depth_image_, CV_8UC1);
    cv::applyColorMap(
      depth_image_,
      depth_image_,
      cv::COLORMAP_JET);
    cv::imshow("Depth Image", depth_image_);
    cv::imshow("Color Image", color_image_);
  }
  pipeline_->stop();
  std::cout << "Ended to capture images" << std::endl;
}
