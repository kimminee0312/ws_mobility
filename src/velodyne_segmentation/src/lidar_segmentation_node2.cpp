#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/dbscan.h>   // DBSCAN 헤더 추가
#include <pcl/common/common.h>

using std::placeholders::_1;
using PointT = pcl::PointXYZI;

struct BBox {
  Eigen::Vector4f min_pt, max_pt;
  Eigen::Vector3f size() const { return (max_pt - min_pt).head<3>(); }
  Eigen::Vector3f center() const { return (0.5f*(max_pt+min_pt)).head<3>(); }
};

class LidarSegNode : public rclcpp::Node {
public:
  LidarSegNode() : Node("lidar_segmentation_node2") {
    // Parameters
    input_topic_       = this->declare_parameter<std::string>("input_topic", "/points_raw/downsampled");
    voxel_leaf_        = this->declare_parameter<double>("voxel_leaf", 0.0);
    z_min_             = this->declare_parameter<double>("z_min", -2.0);
    z_max_             = this->declare_parameter<double>("z_max",  2.0);
    dist_thresh_       = this->declare_parameter<double>("ground_dist_thresh", 0.10);
    eps_angle_deg_     = this->declare_parameter<double>("ground_eps_angle_deg", 10.0);
    cluster_tol_       = this->declare_parameter<double>("cluster_tolerance", 0.20); // DBSCAN eps
    cluster_min_size_  = this->declare_parameter<int>("cluster_min_size", 0);
    cluster_max_size_  = this->declare_parameter<int>("cluster_max_size", 5000);
    min_pts_           = this->declare_parameter<int>("dbscan_min_pts", 5); // DBSCAN minPts

    // Ground segmentation on/off
    use_ground_seg_    = this->declare_parameter<bool>("use_ground_segmentation", true);

    // Cone rules
    cone_min_height_   = this->declare_parameter<double>("cone_min_height", 0.3);
    cone_max_height_   = this->declare_parameter<double>("cone_max_height", 1.2);
    cone_max_xy_       = this->declare_parameter<double>("cone_max_xy_diameter", 0.3);
    cone_height_ratio_ = this->declare_parameter<double>("cone_height_over_xy_min_ratio", 1.0);
    cone_min_points_   = this->declare_parameter<int>("cone_min_points", 20);
    cone_max_points_   = this->declare_parameter<int>("cone_max_points", 40);
    use_intensity_     = this->declare_parameter<bool>("cone_use_intensity", false);
    cone_intensity_min_= this->declare_parameter<double>("cone_intensity_min", 5.0);
    cone_intensity_max_= this->declare_parameter<double>("cone_intensity_max", 255.0);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&LidarSegNode::cbCloud, this, _1));

    pub_ground_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_cloud", 1);
    pub_nonground_= this->create_publisher<sensor_msgs::msg::PointCloud2>("nonground_cloud", 1);
    pub_cone_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_cloud", 1);
    pub_obj_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("objects_markers", 1);
    pub_cone_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cone_markers", 1);

    RCLCPP_INFO(get_logger(), "velodyne_segmentation started. Subscribing: %s (ground segmentation: %s)",
                input_topic_.c_str(), use_ground_seg_ ? "ON" : "OFF");
  }

private:
  void cbCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *cloud);
    RCLCPP_INFO(get_logger(), "▶ Input cloud size: %zu", cloud->size());
    if (cloud->empty()) return;

    // ROI + Downsample
    pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>());
    {
      pcl::PassThrough<PointT> pass; pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(z_min_, z_max_);
      pass.filter(*cloud_f);
      RCLCPP_INFO(get_logger(), "▶ After Z filter: %zu", cloud_f->size());

      if (voxel_leaf_ > 1e-6) {
        pcl::VoxelGrid<PointT> vg; vg.setInputCloud(cloud_f);
        vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        vg.filter(*cloud_f);
        RCLCPP_INFO(get_logger(), "▶ After VoxelGrid: %zu", cloud_f->size());
      }
    }
    if (cloud_f->empty()) return;

    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr nonground(new pcl::PointCloud<PointT>());

    if (!use_ground_seg_) {
      *nonground = *cloud_f;
      RCLCPP_INFO(get_logger(), "▶ Skipping ground segmentation. Nonground size: %zu", nonground->size());
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

      RCLCPP_INFO(get_logger(), "▶ Ground: %zu, Nonground: %zu", ground->size(), nonground->size());
    }

    // === DBSCAN 클러스터링 ===
    std::vector<pcl::PointIndices> cluster_indices;
    if (!nonground->empty()) {
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      tree->setInputCloud(nonground);

      pcl::DBSCANKdtreeCluster<PointT> dbs;
      dbs.setCorePointMinPts(min_pts_);        // minPts
      dbs.setClusterTolerance(cluster_tol_);   // eps
      dbs.setMinClusterSize(cluster_min_size_);
      dbs.setMaxClusterSize(cluster_max_size_);
      dbs.setSearchMethod(tree);
      dbs.setInputCloud(nonground);
      dbs.extract(cluster_indices);
    }
    RCLCPP_INFO(get_logger(), "▶ DBSCAN cluster count: %zu", cluster_indices.size());

    // 이후 콘 후보 판별 부분 (동일)
    visualization_msgs::msg::MarkerArray obj_markers, cone_markers;
    pcl::PointCloud<PointT>::Ptr cone_cloud(new pcl::PointCloud<PointT>());
    int id = 0, cone_id = 0;

    for (const auto &indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
      cluster->reserve(indices.indices.size());
      double intensity_sum = 0.0;

      for (int idx : indices.indices) {
        cluster->push_back((*nonground)[idx]);
        intensity_sum += (*nonground)[idx].intensity;
      }
      const double intensity_mean = cluster->empty() ? 0.0 : (intensity_sum / cluster->size());
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

      if (use_intensity_) {
        cone_like = cone_like && (intensity_mean >= cone_intensity_min_) && (intensity_mean <= cone_intensity_max_);
      }

      if (cone_like) {
        *cone_cloud += *cluster;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = msg->header.frame_id;
        m.header.stamp = this->now();
        m.ns = "cones";
        m.id = cone_id++;
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = center.x();
        m.pose.position.y = center.y();
        m.pose.position.z = box.min_pt.z() + dz/2.0f;
        m.pose.orientation.w = 1.0;
        m.scale.x = std::max(0.1, std::min(cone_max_xy_, xy));
        m.scale.y = m.scale.x;
        m.scale.z = std::max(0.2, std::min(cone_max_height_, dz));
        m.color.r = 1.0f; m.color.g = 0.5f; m.color.b = 0.0f; m.color.a = 0.9f;
        m.lifetime = rclcpp::Duration::from_seconds(0.2);
        cone_markers.markers.push_back(m);
        RCLCPP_INFO(get_logger(), "     -> Cone candidate detected!");
      }
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
    m.lifetime = rclcpp::Duration::from_seconds(0.2);
    return m;
  }

  // Params
  std::string input_topic_;
  double voxel_leaf_, z_min_, z_max_;
  double dist_thresh_, eps_angle_deg_;
  double cluster_tol_; int cluster_min_size_, cluster_max_size_;
  int min_pts_;
  bool use_ground_seg_;
  double cone_min_height_, cone_max_height_, cone_max_xy_, cone_height_ratio_;
  int cone_min_points_, cone_max_points_;
  bool use_intensity_; double cone_intensity_min_, cone_intensity_max_;

  // ROS I/O
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_, pub_nonground_, pub_cone_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_obj_markers_, pub_cone_markers_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarSegNode>());
  rclcpp::shutdown();
  return 0;
}
