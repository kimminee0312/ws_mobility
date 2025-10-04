#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_array.hpp>

using PointT = pcl::PointXYZ;

class ParkingNode : public rclcpp::Node {
public:
  ParkingNode() : Node("parking_node") {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/points_raw/downsampled",
      rclcpp::SensorDataQoS(),
      std::bind(&ParkingNode::callback, this, std::placeholders::_1));

    pub_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cone_clusters", 1);

    pub_colored_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cone_colored_points", 10);

    pub_parking_entrance_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "/cone_parking_entrance", 10);

  }

private:
  float prev_first_y_ = NAN;

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 기존 마커 지우기 (잔상 방지)
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.push_back(clear_marker);
    pub_marker_->publish(clear_markers);

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    // KD-Tree for clustering
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    // Euclidean cluster extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.4);
    ec.setMinClusterSize(3);
    ec.setMaxClusterSize(40);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray markers;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<Eigen::Vector3f> centroids;

    int id = 0;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    // 클러스터 → 중심점 추출
    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
      for (int idx : indices.indices)
        cluster->push_back((*cloud)[idx]);

      // Bounding box
      Eigen::Vector4f min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);
      float height = max_pt.z() - min_pt.z();
      float width  = max_pt.x() - min_pt.x();
      float depth  = max_pt.y() - min_pt.y();

      // 라바콘 조건 필터
      if (height < 0.2 || height > 0.9) continue;
      if (width > 0.4 || depth > 0.4) continue;

      // 랜덤 색상
      uint8_t r = dist(rng), g = dist(rng), b = dist(rng);
      for (const auto& p : cluster->points) {
        pcl::PointXYZRGB pt;
        pt.x = p.x; pt.y = p.y; pt.z = p.z;
        pt.r = r; pt.g = g; pt.b = b;
        colored_cloud->points.push_back(pt);
      }

      // 중심점
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster, centroid);
      centroids.push_back(centroid.head<3>());

      // Marker (파란색 원기둥)
      visualization_msgs::msg::Marker m;
      m.header = msg->header;
      m.ns = "cones";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CYLINDER;   // ✅ 원기둥
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = centroid[0];
      m.pose.position.y = centroid[1];
      m.pose.position.z = centroid[2];
      m.scale.x = 0.3;   // 원기둥 지름
      m.scale.y = 0.3;
      m.scale.z = 0.5;   // 원기둥 높이
      m.color.r = 0.0;   // ✅ 파란색
      m.color.g = 0.0;
      m.color.b = 1.0;
      m.color.a = 1.0;
      // m.lifetime = rclcpp::Duration(0, 200000);
      markers.markers.push_back(m);
    }

    if (centroids.size() < 2) {
      pub_marker_->publish(markers);
      return;
    }

    // 1️⃣ 첫 번째 점 (원점과 가장 가까운 점)
    // to do ========================== 가장 가까운 점 재정의 필요 ===========================
    int first_idx = -1;
    float min_dist = 1e9;
    for (int i = 0; i < (int)centroids.size(); i++) {
      float d = std::hypot(centroids[i][0], centroids[i][1]);
      if (d < min_dist ) {
        min_dist = d;
        first_idx = i;
      }
    }

    std::vector<int> selected;
    if (first_idx >= 0) {
      // 🔹 y좌표 튐 보정 로직
      if (!std::isnan(prev_first_y_)) {
        if (std::fabs(centroids[first_idx][1] - prev_first_y_) > 0.2f) {
          RCLCPP_WARN(this->get_logger(), "First point y jump detected, reverting to previous y");
          centroids[first_idx][1] = prev_first_y_;  // 이전 y좌표 유지
        }
      }
      prev_first_y_ = centroids[first_idx][1];

      selected.push_back(first_idx);
    }

    // 빨간색 첫 점
    visualization_msgs::msg::Marker first_m;
    first_m.header = msg->header;
    first_m.ns = "fitting_start";
    first_m.id = 0;
    first_m.type = visualization_msgs::msg::Marker::SPHERE;
    first_m.action = visualization_msgs::msg::Marker::ADD;
    first_m.pose.position.x = centroids[first_idx][0];
    first_m.pose.position.y = centroids[first_idx][1];
    first_m.pose.position.z = centroids[first_idx][2];
    first_m.scale.x = 0.3;
    first_m.scale.y = 0.3;
    first_m.scale.z = 0.3;
    first_m.color.r = 1.0;
    first_m.color.g = 0.0;
    first_m.color.b = 0.0;
    first_m.color.a = 1.0;
    // first_m.lifetime = rclcpp::Duration(0, 200000);
    markers.markers.push_back(first_m);

    // 2️⃣ 두 번째 점 (센서 좌표계에서 찾기)
    int second_idx = -1;
    float best_dist = 4.0;
    float best_y_diff = 1e9;
    float min_angle_rad = 45*M_PI/180;

    for (int i = 0; i < (int)centroids.size(); i++) {
      if (i == first_idx) continue;
      float dx = centroids[i][0] - centroids[first_idx][0];
      float dy = centroids[i][1] - centroids[first_idx][1];
      float dz = centroids[i][2] - centroids[first_idx][2];
      float d = std::sqrt(dx*dx + dy*dy + dz*dz);
      float angle = atan2(dy, dx);

      if (d <= 4.0 && fabs(dy) <= 0.5 ) {
        if (d < best_dist || (fabs(d - best_dist) < 1e-3 && fabs(dy) < best_y_diff)) {
          if (fabs(angle) <= min_angle_rad){
            best_dist = d;
            best_y_diff = fabs(dy);
            min_angle_rad = angle;
            second_idx = i;
          }
        }
      }
    }

    if (second_idx == -1) {
      pub_marker_->publish(markers);
      return;
    }

    selected.push_back(second_idx);

    // 3️⃣ 이후 점: 로컬 좌표계 기반 탐색
    int current = second_idx;
    Eigen::Vector3f p0 = centroids[first_idx];
    Eigen::Vector3f p1 = centroids[second_idx];
    Eigen::Vector3f x_axis = (p1 - p0).normalized();
    Eigen::Vector3f z_axis(0,0,1);
    Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();
    z_axis = x_axis.cross(y_axis).normalized(); // 정규직교 보정
    Eigen::Matrix3f R;
    R.col(0) = x_axis;
    R.col(1) = y_axis;
    R.col(2) = z_axis;

    while (true) {
      int next_idx = -1;
      float best_local_x = 4.0;
      float best_y_diff2 = 1e9;

      for (int i = 0; i < (int)centroids.size(); i++) {
        if (std::find(selected.begin(), selected.end(), i) != selected.end())
          continue;

        Eigen::Vector3f vec = centroids[i] - centroids[current];
        Eigen::Vector3f local = R.transpose() * vec;

        float lx = local[0];
        float ly = local[1];

        if (lx > 0 && lx <= 4.0 && fabs(ly) <= 0.2) {
          if (lx < best_local_x || (fabs(lx - best_local_x) < 1e-3 && fabs(ly) < best_y_diff2)) {
            best_local_x = lx;
            best_y_diff2 = fabs(ly);
            next_idx = i;
          }
        }
      }

      if (next_idx == -1) break;
      selected.push_back(next_idx);
      current = next_idx;
    }

        // 5️⃣ 가장 먼 두 점 찾아서 원기둥 마커 추가
    if (selected.size() >= 2) {
      float max_dist = -1.0;
      int idx1 = -1, idx2 = -1;

    for (size_t i=0; i<selected.size()-1; i++) {
        float d = (centroids[selected[i]] - centroids[selected[i+1]]).norm();
        if (d > max_dist) {
            max_dist = d;
            if (max_dist >=2.8) {
                idx1 = selected[i];
                idx2 = selected[i+1];
            }
        }
    }
      

      if (idx1 != -1 && idx2 != -1) {
        visualization_msgs::msg::Marker cyl1, cyl2;

        // 첫 번째 원기둥
        cyl1.header = msg->header;
        cyl1.ns = "farthest_points";
        cyl1.id = 1001;
        cyl1.type = visualization_msgs::msg::Marker::CYLINDER;
        cyl1.action = visualization_msgs::msg::Marker::ADD;
        cyl1.pose.position.x = centroids[idx1][0];
        cyl1.pose.position.y = centroids[idx1][1];
        cyl1.pose.position.z = centroids[idx1][2];
        cyl1.scale.x = 0.4; // 원기둥 지름
        cyl1.scale.y = 0.4;
        cyl1.scale.z = 0.6; // 높이
        cyl1.color.r = 1.0;
        cyl1.color.g = 0.0;
        cyl1.color.b = 0.0;
        cyl1.color.a = 1.0;
        // cyl1.lifetime = rclcpp::Duration(0, 200000);
        markers.markers.push_back(cyl1);

        // 두 번째 원기둥
        cyl2 = cyl1;  // 복사 후 위치만 바꿈
        cyl2.id = 1002;
        cyl2.pose.position.x = centroids[idx2][0];
        cyl2.pose.position.y = centroids[idx2][1];
        cyl2.pose.position.z = centroids[idx2][2];
        // cyl2.lifetime = rclcpp::Duration(0, 200000);
        markers.markers.push_back(cyl2);

        // ✅ 좌표 및 거리 출력
        float dx = centroids[idx2][0] - centroids[idx1][0];
        float dy = centroids[idx2][1] - centroids[idx1][1];
        float dz = centroids[idx2][2] - centroids[idx1][2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

        RCLCPP_INFO(this->get_logger(),
                    "Red Cylinders:\n P1=(%.2f, %.2f, %.2f)\n P2=(%.2f, %.2f, %.2f)\n Distance=%.2f m",
                    centroids[idx1][0], centroids[idx1][1], centroids[idx1][2],
                    centroids[idx2][0], centroids[idx2][1], centroids[idx2][2],
                    dist);

        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header = msg->header;
        geometry_msgs::msg::Pose p1, p2;
        p1.position.x = centroids[idx1][0];
        p1.position.y = centroids[idx1][1];
        p1.position.z = centroids[idx1][2];
        p2.position.x = centroids[idx2][0];
        p2.position.y = centroids[idx2][1];
        p2.position.z = centroids[idx2][2];
        pose_array.poses.push_back(p1);
        pose_array.poses.push_back(p2);
        pub_parking_entrance_->publish(pose_array);
      }
    }

    // 4️⃣ 선택된 점들 라인으로 표시
    if (selected.size() >= 2) {
      visualization_msgs::msg::Marker line;
      line.header = msg->header;
      line.ns = "fitting_line";
      line.id = 0;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.2;
      line.color.r = 0.53;
      line.color.g = 0.81;
      line.color.b = 0.92;
      line.color.a = 0.3;
      // line.lifetime = rclcpp::Duration(0, 200000);

      for (int idx : selected) {
        geometry_msgs::msg::Point p;
        p.x = centroids[idx][0];
        p.y = centroids[idx][1];
        p.z = centroids[idx][2];
        line.points.push_back(p);
      }

      markers.markers.push_back(line);
    }

    // Publish
    pub_marker_->publish(markers);

    sensor_msgs::msg::PointCloud2 color_msg;
    pcl::toROSMsg(*colored_cloud, color_msg);
    color_msg.header = msg->header;
    pub_colored_cloud_->publish(color_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_cloud_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_parking_entrance_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingNode>());
  rclcpp::shutdown();
  return 0;
}