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

  color_intrinsics_matrix_ = cv::Mat::eye(3, 3, CV_64F);
  color_intrinsics_matrix_.at<double>(0, 0) = color_intrinsics.fx;
  color_intrinsics_matrix_.at<double>(0, 2) = color_intrinsics.ppx;
  color_intrinsics_matrix_.at<double>(1, 1) = color_intrinsics.fy;
  color_intrinsics_matrix_.at<double>(1, 2) = color_intrinsics.ppy;
  color_intrinsics_matrix_.at<double>(2, 2) = 1.0f;

  std::vector<float> distortion_coeffs(color_intrinsics.coeffs, color_intrinsics.coeffs + 5);
  distortion_matrix_ = cv::Mat(1, 5, CV_32F, distortion_coeffs.data()).clone();

  std::cout << "Intrensic matrix: " << color_intrinsics_matrix_ << std::endl;
  std::cout << "Distortion matrix: " << distortion_matrix_ << std::endl;

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

    depth_image_.convertTo(depth_image_, CV_32FC1, 0.001);

    estimate_depth(depth_image_, color_image_);

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

bool realsense_reader::ImageHandler::estimate_depth(
  const cv::Mat & depth_image,
  cv::Mat & color_image)
{
  cv::aruco::Dictionary aruco_dictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  cv::aruco::DetectorParameters detector_params;
  cv::aruco::ArucoDetector aruco_detector(aruco_dictionary, detector_params);

  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
  aruco_detector.detectMarkers(color_image_, marker_corners, marker_ids, rejected_candidates);

  if (!marker_ids.empty()) {
    cv::aruco::drawDetectedMarkers(color_image_, marker_corners, marker_ids);
  } else {
    std::cout << "Failed to detect markers" << std::endl;
    return false;
  }

  float markerLength = 0.199;
  std::vector<cv::Point3f> object_points = {
    cv::Point3f(0, 0, 0),
    cv::Point3f(markerLength, 0, 0),
    cv::Point3f(markerLength, markerLength, 0),
    cv::Point3f(0, markerLength, 0)};

  cv::Mat world_to_points(4, 4, CV_64F);

  for (size_t i = 0; i < object_points.size(); ++i) {
    world_to_points.at<double>(0, i) = object_points[i].x;
    world_to_points.at<double>(1, i) = object_points[i].y;
    world_to_points.at<double>(2, i) = object_points[i].z;
    world_to_points.at<double>(3, i) = 1.0;
  }

  for (size_t i = 0; i < marker_ids.size(); ++i) {
    cv::Vec3d rotation, translation;
    cv::solvePnP(object_points, marker_corners[i], color_intrinsics_matrix_, distortion_matrix_, rotation, translation);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation, rotation_matrix);

    cv::Mat projection_matrix, camera_to_world;
    cv::hconcat(rotation_matrix, translation, projection_matrix);
    cv::vconcat(projection_matrix, cv::Mat(cv::Vec4d(0, 0, 0, 1)).t(), camera_to_world);

    std::cout << "Marker ID: " << marker_ids[i] << std::endl;
    std::cout << "Rotation Vector: " << rotation << std::endl;
    std::cout << "Translation Vector: " << translation << std::endl;
    std::cout << "Camera Position (World Coordinates): " << std::endl;
    std::cout << camera_to_world.inv() << std::endl;

    cv::Mat camera_to_points = camera_to_world * world_to_points;
    std::vector<float> aruco_depth = {
      static_cast<float>(camera_to_points.at<double>(2, 0)),
      static_cast<float>(camera_to_points.at<double>(2, 1)),
      static_cast<float>(camera_to_points.at<double>(2, 2)),
      static_cast<float>(camera_to_points.at<double>(2, 3))};
    std::vector<float> rs_depth = {
      depth_image_.at<float>(static_cast<int>(marker_corners[i][0].y), static_cast<int>(marker_corners[i][0].x)),
      depth_image_.at<float>(static_cast<int>(marker_corners[i][1].y), static_cast<int>(marker_corners[i][1].x)),
      depth_image_.at<float>(static_cast<int>(marker_corners[i][2].y), static_cast<int>(marker_corners[i][2].x)),
      depth_image_.at<float>(static_cast<int>(marker_corners[i][3].y), static_cast<int>(marker_corners[i][3].x))};

    for (size_t i = 0; i < aruco_depth.size(); i++)  {
      std::cout << "Aruco Depth : " << aruco_depth[i] << " / rs dpeth : " << rs_depth[i] << " / diff : " << std::fabs(aruco_depth[i] - rs_depth[i]) << std::endl;
    }
  }

  return true;
}
