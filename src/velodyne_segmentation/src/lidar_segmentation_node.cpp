#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;
using PointT = pcl::PointXYZ;

struct BBox {
  Eigen::Vector4f min_pt, max_pt;
  Eigen::Vector3f size() const { return (max_pt - min_pt).head<3>(); }
  Eigen::Vector3f center() const { return (0.5f * (max_pt + min_pt)).head<3>(); }
};

class LidarSegNode : public rclcpp::Node {
public:
  LidarSegNode() : Node("lidar_segmentation_node") {
    // Parameters
    input_topic_       = this->declare_parameter<std::string>("input_topic", "/points_raw/downsampled");
    voxel_leaf_        = this->declare_parameter<double>("voxel_leaf", 0.0);
    z_min_             = this->declare_parameter<double>("z_min", -2.0);
    z_max_             = this->declare_parameter<double>("z_max",  2.0);
    dist_thresh_       = this->declare_parameter<double>("ground_dist_thresh", 0.10);
    eps_angle_deg_     = this->declare_parameter<double>("ground_eps_angle_deg", 10.0);
    cluster_tol_       = this->declare_parameter<double>("cluster_tolerance", 0.12);
    cluster_min_size_  = this->declare_parameter<int>("cluster_min_size", 0);
    cluster_max_size_  = this->declare_parameter<int>("cluster_max_size", 5000);

    // Ground segmentation on/off
    use_ground_seg_    = this->declare_parameter<bool>("use_ground_segmentation", true);

    // Cone rules
    cone_min_height_   = this->declare_parameter<double>("cone_min_height", 0.3);
    cone_max_height_   = this->declare_parameter<double>("cone_max_height", 0.9);
    cone_max_xy_       = this->declare_parameter<double>("cone_max_xy_diameter", 0.28);
    cone_height_ratio_ = this->declare_parameter<double>("cone_height_over_xy_min_ratio", 1.0);
    cone_min_points_   = this->declare_parameter<int>("cone_min_points", 10);
    cone_max_points_   = this->declare_parameter<int>("cone_max_points", 35);

    // Subscribers & Publishers
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LidarSegNode::cbCloud, this, _1));

    pub_ground_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_cloud", 1);
    pub_nonground_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("nonground_cloud", 1);
    pub_cone_         = this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_cloud", 1);
    pub_obj_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("objects_markers", 1);
    pub_cone_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cone_markers", 1);
    pub_cone_centers_ = this->create_publisher<geometry_msgs::msg::PoseArray>("cone_centers", 1);

    // CSV 파일 초기화
    std::string csv_path = "csv/cone_centers.csv";
    csv_file_.open(csv_path, std::ios::out | std::ios::trunc);
    if (csv_file_.is_open()) {
      csv_file_ << "time,x,y,z,side\n";
      csv_file_.flush();
      RCLCPP_INFO(this->get_logger(), "CSV logging to: %s", csv_path.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file at %s", csv_path.c_str());
    }
  }

  ~LidarSegNode() {
    if (csv_file_.is_open()) {
      csv_file_.close();
    }
  }

private:
  void cbCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    // ROI + Downsample
    pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>());
    {
      pcl::PassThrough<PointT> pass; pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(z_min_, z_max_);
      pass.filter(*cloud_f);

      if (voxel_leaf_ > 1e-6) {
        pcl::VoxelGrid<PointT> vg; vg.setInputCloud(cloud_f);
        vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        vg.filter(*cloud_f);
      }
    }
    if (cloud_f->empty()) return;

    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr nonground(new pcl::PointCloud<PointT>());

    if (!use_ground_seg_) {
      *nonground = *cloud_f;
    } else {
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(dist_thresh_);
      seg.setMaxIterations(100);
      Eigen::Vector3f axis(0.0f, 0.0f, -1.0f);
      seg.setAxis(axis);
      seg.setEpsAngle(static_cast<float>(eps_angle_deg_ * M_PI/180.0));

      pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());
      pcl::ModelCoefficients::Ptr ground_coeff(new pcl::ModelCoefficients());
      seg.setInputCloud(cloud_f);
      seg.segment(*ground_inliers, *ground_coeff);

      pcl::ExtractIndices<PointT> ei;
      ei.setInputCloud(cloud_f);
      ei.setIndices(ground_inliers);
      ei.setNegative(false);
      ei.filter(*ground);
      ei.setNegative(true);
      ei.filter(*nonground);
    }

    // 클러스터링
    std::vector<pcl::PointIndices> cluster_indices;
    if (!nonground->empty()) {
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      tree->setInputCloud(nonground);
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance(cluster_tol_);
      ec.setMinClusterSize(cluster_min_size_);
      ec.setMaxClusterSize(cluster_max_size_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(nonground);
      ec.extract(cluster_indices);
    }

    visualization_msgs::msg::MarkerArray obj_markers, cone_markers;
    pcl::PointCloud<PointT>::Ptr cone_cloud(new pcl::PointCloud<PointT>());
    std::vector<geometry_msgs::msg::Pose> all_cones;

    int id = 0, cone_id = 0;
    auto now = std::chrono::system_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    for (const auto &indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
      cluster->reserve(indices.indices.size());

      for (int idx : indices.indices) {
        if (idx >= 0 && idx < (int)nonground->size()) {
          cluster->push_back((*nonground)[idx]);
        }
      }
      if (cluster->empty()) continue;

      BBox box = computeBBox(*cluster);
      const auto size = box.size();
      const auto center = box.center();

      obj_markers.markers.push_back(makeBoxMarker("objects", id++, msg->header.frame_id,
                                                  center, size, 0.0f, 0.6f, 1.0f, 0.25f));

      const double dx = size.x(), dy = size.y(), dz = size.z();
      const double xy = std::max(dx, dy);
      bool cone_like = (dz >= cone_min_height_) && (dz <= cone_max_height_) &&
                       (xy <= cone_max_xy_) &&
                       (dz / std::max(0.05, xy) >= cone_height_ratio_) &&
                       (static_cast<int>(cluster->size()) >= cone_min_points_) &&
                       (static_cast<int>(cluster->size()) <= cone_max_points_);

      if (cone_like) {
        *cone_cloud += *cluster;

        geometry_msgs::msg::Pose p;
        p.position.x = center.x();
        p.position.y = center.y();
        p.position.z = center.z();
        p.orientation.w = 1.0;
        all_cones.push_back(p);
      }
    }

    // --- 좌/우 재분류: X축 기준 정렬 후 짝짓기 ---
    std::sort(all_cones.begin(), all_cones.end(),
              [](const geometry_msgs::msg::Pose& a, const geometry_msgs::msg::Pose& b){
                return a.position.x < b.position.x;
              });

    std::vector<geometry_msgs::msg::Pose> left_cones, right_cones;
    for (size_t i=0; i+1<all_cones.size(); i+=2) {
      auto &p1 = all_cones[i];
      auto &p2 = all_cones[i+1];

      if (p1.position.y > p2.position.y) {
        left_cones.push_back(p1);
        right_cones.push_back(p2);
      } else {
        left_cones.push_back(p2);
        right_cones.push_back(p1);
      }
    }

    // 메시지 만들기
    geometry_msgs::msg::PoseArray cone_centers_msg;
    cone_centers_msg.header = msg->header;
    cone_centers_msg.poses.insert(cone_centers_msg.poses.end(), left_cones.begin(), left_cones.end());
    cone_centers_msg.poses.insert(cone_centers_msg.poses.end(), right_cones.begin(), right_cones.end());

    // CSV 저장 + Marker 발행
    int cone_id_csv = 0;
    for (auto& p : left_cones) {
      if (csv_file_.is_open()) {
        csv_file_ << time_ms << "," << p.position.x << "," << p.position.y << "," << p.position.z << ",LEFT\n";
        csv_file_.flush();
      }
      visualization_msgs::msg::Marker m;
      m.header.frame_id = msg->header.frame_id;
      m.header.stamp = this->now();
      m.ns = "cones";
      m.id = cone_id_csv++;
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = p;
      m.scale.x = 0.25; m.scale.y = 0.25; m.scale.z = 0.5;
      m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 0.9f;
      cone_markers.markers.push_back(m);
    }
    for (auto& p : right_cones) {
      if (csv_file_.is_open()) {
        csv_file_ << time_ms << "," << p.position.x << "," << p.position.y << "," << p.position.z << ",RIGHT\n";
        csv_file_.flush();
      }
      visualization_msgs::msg::Marker m;
      m.header.frame_id = msg->header.frame_id;
      m.header.stamp = this->now();
      m.ns = "cones";
      m.id = cone_id_csv++;
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = p;
      m.scale.x = 0.25; m.scale.y = 0.25; m.scale.z = 0.5;
      m.color.r = 0.0f; m.color.g = 0.0f; m.color.b = 1.0f; m.color.a = 0.9f;
      cone_markers.markers.push_back(m);
    }

    // 퍼블리시
    sensor_msgs::msg::PointCloud2 ground_msg, nonground_msg, cone_msg;
    pcl::toROSMsg(*ground, ground_msg);
    pcl::toROSMsg(*nonground, nonground_msg);
    pcl::toROSMsg(*cone_cloud, cone_msg);
    ground_msg.header = nonground_msg.header = cone_msg.header = msg->header;

    pub_ground_->publish(ground_msg);
    pub_nonground_->publish(nonground_msg);
    pub_cone_->publish(cone_msg);
    pub_obj_markers_->publish(obj_markers);
    pub_cone_markers_->publish(cone_markers);

    if (!cone_centers_msg.poses.empty()) {
      pub_cone_centers_->publish(cone_centers_msg);
    }
  }

  static BBox computeBBox(const pcl::PointCloud<PointT>& cloud) {
    BBox box;
    pcl::getMinMax3D(cloud, box.min_pt, box.max_pt);
    return box;
  }

  static visualization_msgs::msg::Marker makeBoxMarker(
      const std::string& ns, int id, const std::string& frame,
      const Eigen::Vector3f& center, const Eigen::Vector3f& size,
      float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame;
    m.header.stamp = rclcpp::Clock().now();
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = center.x();
    m.pose.position.y = center.y();
    m.pose.position.z = center.z();
    m.pose.orientation.w = 1.0;
    m.scale.x = std::max(0.05f, size.x());
    m.scale.y = std::max(0.05f, size.y());
    m.scale.z = std::max(0.05f, size.z());
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    m.lifetime.sec = 0;
    m.lifetime.nanosec = 200000000;
    return m;
  }

  // Params
  std::string input_topic_;
  double voxel_leaf_, z_min_, z_max_;
  double dist_thresh_, eps_angle_deg_;
  double cluster_tol_; int cluster_min_size_, cluster_max_size_;
  bool use_ground_seg_;
  double cone_min_height_, cone_max_height_, cone_max_xy_, cone_height_ratio_;
  int cone_min_points_, cone_max_points_;

  // CSV 파일
  std::ofstream csv_file_;

  // ROS I/O
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_, pub_nonground_, pub_cone_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_obj_markers_, pub_cone_markers_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_cone_centers_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarSegNode>());
  rclcpp::shutdown();
  return 0;
}
