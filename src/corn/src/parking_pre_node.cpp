#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <iomanip>
#include <array>
#include <sstream>
#include <algorithm>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl/filters/passthrough.h>
#include <limits>

// TF2
#include <tf2_ros/transform_broadcaster.h>

using PointT = pcl::PointXYZ;

class ParkingPreNode : public rclcpp::Node {
public:
  ParkingPreNode() : Node("parking_pre_node") {
    sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/points_raw/downsampled", rclcpp::SensorDataQoS(),
      std::bind(&ParkingPreNode::cloudCallback, this, std::placeholders::_1));

    pub_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/closest_cone_markers", 10);

    pub_marker_groups_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cone_group_markers", 10);

    pub_grid_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/cone_grid", 10);

    pub_cluster_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/cone_clusters_colored", 10);

    // z-cut 이후 사용 포인트들 시각화(deg)
    pub_deg_zcloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/points_deg_zfiltered_colored", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    has_initialized_frame_ = false;
    RCLCPP_INFO(this->get_logger(), "✅ ParkingPreNode started.");
  }

private:
  // --- state ---
  bool has_initialized_frame_;
  Eigen::Vector3f origin_;
  Eigen::Matrix3f R_;

  // for ns/id reuse & selective DELETE
  int last_cone_count_ = 0;
  int last_group_x_count_ = 0;
  int last_group_y_count_ = 0;

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // PCL 변환
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    // ✅ z > -0.21 m 필터링 (지면 근처 제거)
    pcl::PointCloud<PointT>::Ptr cloud_z(new pcl::PointCloud<PointT>);
    {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(-0.21f, std::numeric_limits<float>::max());
      pass.filter(*cloud_z);
    }
    if (cloud_z->empty()) return;

    // === z-cut을 통과한 포인트만 보라색으로 퍼블리시 (deg) ===
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr zcolored(new pcl::PointCloud<pcl::PointXYZRGB>);
      zcolored->points.reserve(cloud_z->points.size());
      for (const auto& p : cloud_z->points) {
        pcl::PointXYZRGB prgb;
        prgb.x = p.x; prgb.y = p.y; prgb.z = p.z;
        prgb.r = 255; prgb.g = 0; prgb.b = 255;  // 보라색
        zcolored->points.push_back(prgb);
      }
      zcolored->width  = static_cast<uint32_t>(zcolored->points.size());
      zcolored->height = 1;
      zcolored->is_dense = false;
      sensor_msgs::msg::PointCloud2 zmsg;
      pcl::toROSMsg(*zcolored, zmsg);
      zmsg.header = msg->header;
      pub_deg_zcloud_->publish(zmsg);
    }

    // --- 클러스터링 (필터된 점군으로) ---
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_z);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(20000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_z);
    ec.extract(cluster_indices);

    std::vector<Eigen::Vector3f> centroids;
    pcl::PointCloud<PointT>::Ptr cloud_outer(new pcl::PointCloud<PointT>);

    centroids.reserve(cluster_indices.size());

    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
      cluster->points.reserve(indices.indices.size());
      for (int idx : indices.indices) cluster->push_back((*cloud_z)[idx]);

      // 라바콘 형태 필터 (AABB)
      Eigen::Vector4f min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);
      float height = max_pt.z() - min_pt.z();
      float width  = max_pt.x() - min_pt.x();
      float depth  = max_pt.y() - min_pt.y();
      if (height < 0.1f || height > 1.0f || width > 0.4f || depth > 0.4f) {
        *cloud_outer += *cluster;
        continue;
      }

      // 중심점
      Eigen::Vector4f c;
      pcl::compute3DCentroid(*cluster, c);
      centroids.push_back(c.head<3>());
    }

    // --- 라바콘 2차 검출 ---
    if (!cloud_outer->empty()) {
      pcl::search::KdTree<PointT>::Ptr tree2(new pcl::search::KdTree<PointT>);
      tree2->setInputCloud(cloud_outer);

      std::vector<pcl::PointIndices> cluster_indices2;
      pcl::EuclideanClusterExtraction<PointT> ec2;
      ec2.setClusterTolerance(0.3);   // 더 촘촘하게
      ec2.setMinClusterSize(2);
      ec2.setMaxClusterSize(1200);
      ec2.setSearchMethod(tree2);
      ec2.setInputCloud(cloud_outer);
      ec2.extract(cluster_indices2);

      static const std::array<std::array<uint8_t,3>, 12> PALETTE = {{
        {255,  59,  48}, {255, 149,   0}, {255, 204,   0}, { 52, 199,  89},
        {  0, 199, 190}, { 48, 176, 199}, { 88,  86, 214}, {255,  45,  85},
        {142, 142, 147}, {162, 132,  94}, { 64, 156, 255}, {100, 210, 255}
      }};

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      colored_cloud->is_dense = false;

      size_t cid = 0;
      for (const auto& indices2 : cluster_indices2) {
        const auto& col = PALETTE[cid % PALETTE.size()];
        ++cid;

        pcl::PointCloud<PointT>::Ptr cluster2(new pcl::PointCloud<PointT>);
        cluster2->points.reserve(indices2.indices.size());
        for (int idx : indices2.indices) cluster2->push_back((*cloud_outer)[idx]);

        Eigen::Vector4f min_pt2, max_pt2;
        pcl::getMinMax3D(*cluster2, min_pt2, max_pt2);
        float h2 = max_pt2.z() - min_pt2.z();
        float w2 = max_pt2.x() - min_pt2.x();
        float d2 = max_pt2.y() - min_pt2.y();

        bool cone_like2 =
          (h2 >= 0.1f && h2 <= 1.0f) &&
          (w2 <= 0.5f) &&
          (d2 <= 0.5f);
        if (!cone_like2) continue;

        Eigen::Vector4f c2;
        pcl::compute3DCentroid(*cluster2, c2);
        centroids.push_back(c2.head<3>());

        colored_cloud->points.reserve(colored_cloud->points.size() + cluster2->points.size());
        for (const auto& p : cluster2->points) {
          pcl::PointXYZRGB prgb;
          prgb.x = p.x; prgb.y = p.y; prgb.z = p.z;
          prgb.r = col[0]; prgb.g = col[1]; prgb.b = col[2];
          colored_cloud->points.push_back(prgb);
        }
      }

      if (!colored_cloud->points.empty()) {
        sensor_msgs::msg::PointCloud2 msg_out;
        pcl::toROSMsg(*colored_cloud, msg_out);
        msg_out.header = msg->header;
        msg_out.header.frame_id = msg->header.frame_id;
        pub_cluster_cloud_->publish(msg_out);
      }
    }

    // === 모든 라바콘(라바콘 후보) 원기둥 마커 (ns/id 재사용 + 남는 id DELETE) ===
    visualization_msgs::msg::MarkerArray cone_markers;
    int cone_count = 0;
    for (const auto& c : centroids) {
      visualization_msgs::msg::Marker m;
      m.header = msg->header;
      m.ns = "all_cones";     // 고정 ns
      m.id = cone_count++;    // 0..N-1 재사용
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = c.x();
      m.pose.position.y = c.y();
      m.pose.position.z = c.z();
      m.scale.x = 0.3;   // 지름
      m.scale.y = 0.3;
      m.scale.z = 0.6;   // 높이
      m.color.r = 0.0;
      m.color.g = 0.0;
      m.color.b = 1.0;   // 파란색
      m.color.a = 0.8;
      cone_markers.markers.push_back(m);
    }
    // 사라진 나머지 id 삭제
    for (int i = cone_count; i < last_cone_count_; ++i) {
      visualization_msgs::msg::Marker del;
      del.header = msg->header;
      del.ns = "all_cones";
      del.id = i;
      del.action = visualization_msgs::msg::Marker::DELETE;
      cone_markers.markers.push_back(del);
    }
    last_cone_count_ = cone_count;

    // === p1/p2/arrow 고정 ns/id로 덮어쓰기 ===
    // 차량 원점(0,0,0) 기준 1번(가장 가까운) 찾기
    auto dist2d = [](const Eigen::Vector3f& p){ return std::hypot(p.x(), p.y()); };
    int first_idx = -1;
    float best = 1e9f;
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
      float d = dist2d(centroids[i]);
      if (d < best) { best = d; first_idx = i; }
    }

    visualization_msgs::msg::MarkerArray p12_arr;
    if (first_idx >= 0) {
      const Eigen::Vector3f p1 = centroids[first_idx];
      int second_idx = -1;
      best = 1e9f;
      const float MAX_P12_DIST = 1.5f;

      for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        if (i == first_idx) continue;
        float d_vehicle = std::hypot(centroids[i].x(), centroids[i].y());
        float d12_xy = std::hypot(centroids[i].x() - p1.x(),
                                  centroids[i].y() - p1.y());
        if (d12_xy <= MAX_P12_DIST && d_vehicle < best) {
          best = d_vehicle;
          second_idx = i;
        }
      }

      // p1 marker
      {
        visualization_msgs::msg::Marker m;
        m.header = msg->header; m.ns = "p12"; m.id = 0;
        m.type = visualization_msgs::msg::Marker::SPHERE; m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = p1.x(); m.pose.position.y = p1.y(); m.pose.position.z = p1.z();
        m.scale.x = m.scale.y = m.scale.z = 0.4;
        m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0;
        p12_arr.markers.push_back(m);
      }

      if (second_idx >= 0) {
        const Eigen::Vector3f p2 = centroids[second_idx];

        // p2 marker
        {
          visualization_msgs::msg::Marker m;
          m.header = msg->header; m.ns = "p12"; m.id = 1;
          m.type = visualization_msgs::msg::Marker::SPHERE; m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = p2.x(); m.pose.position.y = p2.y(); m.pose.position.z = p2.z();
          m.scale.x = m.scale.y = m.scale.z = 0.4;
          m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0;
          p12_arr.markers.push_back(m);
        }

        // arrow 1->2
        {
          visualization_msgs::msg::Marker m;
          m.header = msg->header; m.ns = "p12"; m.id = 2;
          m.type = visualization_msgs::msg::Marker::ARROW; m.action = visualization_msgs::msg::Marker::ADD;
          geometry_msgs::msg::Point A,B; A.x=p1.x(); A.y=p1.y(); A.z=p1.z();
          B.x=p2.x(); B.y=p2.y(); B.z=p2.z();
          m.points = {A,B};
          m.scale.x = 0.10; m.scale.y = 0.20; m.scale.z = 0.20;
          m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
          p12_arr.markers.push_back(m);
        }

        // --- 로컬 좌표계 정의 & TF (매 콜백) ---
        Eigen::Vector3f x_axis = (p2 - p1);
        float len = x_axis.norm();
        if (len >= 1e-6f) {
          x_axis /= len;
          const Eigen::Vector3f z_global(0.f, 0.f, 1.f);
          Eigen::Vector3f y_axis = x_axis.cross(z_global);
          if (y_axis.norm() < 1e-6f) y_axis = Eigen::Vector3f(0.f, 1.f, 0.f);
          else y_axis.normalize();
          Eigen::Vector3f z_axis = x_axis.cross(y_axis);
          z_axis.normalize();

          Eigen::Matrix3f R; R.col(0)=x_axis; R.col(1)=y_axis; R.col(2)=z_axis;
          Eigen::Quaternionf q(R); q.normalize();

          geometry_msgs::msg::TransformStamped T;
          T.header = msg->header;                 // 이 시각의 TF로 브로드캐스트
          T.child_frame_id = "cone_local";
          T.transform.translation.x = p1.x();
          T.transform.translation.y = p1.y();
          T.transform.translation.z = p1.z();
          T.transform.rotation.x = q.x();
          T.transform.rotation.y = q.y();
          T.transform.rotation.z = q.z();
          T.transform.rotation.w = q.w();

          // ✅ 매 콜백마다 TF 전송 (동적 프레임)
          tf_broadcaster_->sendTransform(T);

          if (!has_initialized_frame_) {
            origin_ = p1;
            R_ = R;
            has_initialized_frame_ = true;
            RCLCPP_INFO(this->get_logger(), "✅ Local frame initialized (cone_local)");
          }
        } else {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "p1 and p2 are too close; skip TF.");
        }
      } else {
        // 두 번째 점이 없으면 화살표/두 번째 구 마커는 다음 퍼블리시에서 덮어쓰기 되도록 DELETE
        for (int id = 1; id <= 2; ++id) {
          visualization_msgs::msg::Marker del;
          del.header = msg->header; del.ns="p12"; del.id=id; del.action=visualization_msgs::msg::Marker::DELETE;
          p12_arr.markers.push_back(del);
        }
      }
    } else {
      // p1/p2/arrow 모두 삭제
      for (int id = 0; id <= 2; ++id) {
        visualization_msgs::msg::Marker del;
        del.header = msg->header; del.ns="p12"; del.id=id; del.action=visualization_msgs::msg::Marker::DELETE;
        p12_arr.markers.push_back(del);
      }
    }

    // --- 이후부터는 cone_local 기준 변환 (초기화된 경우에만) ---
    std::vector<Eigen::Vector3f> local_cones;
    local_cones.reserve(centroids.size());
    if (has_initialized_frame_) {
      for (const auto& c : centroids) {
        Eigen::Vector3f local = R_.transpose() * (c - origin_);
        local_cones.push_back(local);
      }
    }

    // --- 그룹화 (X좌표 기준) ---
    std::vector<std::vector<Eigen::Vector3f>> groups_x;
    if (!local_cones.empty()) {
      std::vector<Eigen::Vector3f> sorted = local_cones;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.x() < b.x(); });

      std::vector<Eigen::Vector3f> current_group;
      for (size_t i = 0; i < sorted.size(); ++i) {
        if (current_group.empty()) {
          current_group.push_back(sorted[i]);
        } else {
          float diff_x  = std::fabs(sorted[i].x() - current_group.back().x());
          float dist_xy = (sorted[i] - current_group.back()).norm();
          if (diff_x < 0.4f && dist_xy < 1.5f) {
            current_group.push_back(sorted[i]);
          } else {
            groups_x.push_back(current_group);
            current_group.clear();
            current_group.push_back(sorted[i]);
          }
        }
      }
      if (!current_group.empty()) groups_x.push_back(current_group);
    }

    // --- 그룹화 (Y좌표 기준) ---
    std::vector<std::vector<Eigen::Vector3f>> groups_y;
    if (!local_cones.empty()) {
      std::vector<Eigen::Vector3f> sorted = local_cones;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.y() < b.y(); });

      std::vector<Eigen::Vector3f> current_group;
      for (size_t i = 0; i < sorted.size(); ++i) {
        if (current_group.empty()) {
          current_group.push_back(sorted[i]);
        } else {
          float diff_y  = std::fabs(sorted[i].y() - current_group.back().y());
          float dist_xy = (sorted[i] - current_group.back()).norm();
          if (diff_y < 0.4f && dist_xy < 1.5f) {
            current_group.push_back(sorted[i]);
          } else {
            groups_y.push_back(current_group);
            current_group.clear();
            current_group.push_back(sorted[i]);
          }
        }
      }
      if (!current_group.empty()) groups_y.push_back(current_group);
    }

    // --- 그룹화 결과 로그 ---
    RCLCPP_INFO(this->get_logger(), "📌 그룹화 결과 (cone_local 좌표계 기준)");
    RCLCPP_INFO(this->get_logger(), "X축 기준 그룹 수: %zu", groups_x.size());
    for (size_t g = 0; g < groups_x.size(); ++g) {
      std::ostringstream oss; oss << "  Group X" << g << ": ";
      for (const auto& p : groups_x[g]) {
        oss << "(" << std::fixed << std::setprecision(2)
            << p.x() << "," << p.y() << "," << p.z() << ") ";
      }
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }
    RCLCPP_INFO(this->get_logger(), "Y축 기준 그룹 수: %zu", groups_y.size());
    for (size_t g = 0; g < groups_y.size(); ++g) {
      std::ostringstream oss; oss << "  Group Y" << g << ": ";
      for (const auto& p : groups_y[g]) {
        oss << "(" << std::fixed << std::setprecision(2)
            << p.x() << "," << p.y() << "," << p.z() << ") ";
      }
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    // --- 그룹 선 시각화 (ns/id 재사용 + 줄어든 id DELETE) ---
    visualization_msgs::msg::MarkerArray group_markers;

    // X 라인
    for (size_t g = 0; g < groups_x.size(); ++g) {
      visualization_msgs::msg::Marker line;
      line.header = msg->header;
      line.ns = "group_x";                 // 고정 ns
      line.id = static_cast<int>(g);       // 0..Nx-1
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.05;
      line.color.r = 1.0f; line.color.g = 0.5f; line.color.b = 0.0f; line.color.a = 0.95f;

      for (const auto& p_local : groups_x[g]) {
        Eigen::Vector3f global = origin_ + R_ * p_local;
        geometry_msgs::msg::Point P; P.x = global.x(); P.y = global.y(); P.z = global.z();
        line.points.push_back(P);
      }
      if (line.points.size() >= 2) group_markers.markers.push_back(line);
    }
    for (int g = static_cast<int>(groups_x.size()); g < last_group_x_count_; ++g) {
      visualization_msgs::msg::Marker del;
      del.header = msg->header; del.ns="group_x"; del.id=g; del.action=visualization_msgs::msg::Marker::DELETE;
      group_markers.markers.push_back(del);
    }
    last_group_x_count_ = static_cast<int>(groups_x.size());

    // Y 라인
    for (size_t g = 0; g < groups_y.size(); ++g) {
      visualization_msgs::msg::Marker line;
      line.header = msg->header;
      line.ns = "group_y";
      line.id = static_cast<int>(g);
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.05;
      line.color.r = 0.0f; line.color.g = 0.7f; line.color.b = 1.0f; line.color.a = 0.95f;

      for (const auto& p_local : groups_y[g]) {
        Eigen::Vector3f global = origin_ + R_ * p_local;
        geometry_msgs::msg::Point P; P.x = global.x(); P.y = global.y(); P.z = global.z();
        line.points.push_back(P);
      }
      if (line.points.size() >= 2) group_markers.markers.push_back(line);
    }
    for (int g = static_cast<int>(groups_y.size()); g < last_group_y_count_; ++g) {
      visualization_msgs::msg::Marker del;
      del.header = msg->header; del.ns="group_y"; del.id=g; del.action=visualization_msgs::msg::Marker::DELETE;
      group_markers.markers.push_back(del);
    }
    last_group_y_count_ = static_cast<int>(groups_y.size());

    // --- OccupancyGrid 생성/퍼블리시 (탑다운 2D) ---
    if (has_initialized_frame_) {
      const float RES = 0.10f;
      const int   W   = 200;
      const int   H   = 200;
      const float HALF_W = (W * RES) * 0.5f;
      const float HALF_H = (H * RES) * 0.5f;

      nav_msgs::msg::OccupancyGrid grid;
      grid.header = msg->header;
      grid.header.frame_id = "cone_local";
      grid.info.resolution = RES;
      grid.info.width  = W;
      grid.info.height = H;

      // cone_local 좌표계 기준 중앙정렬
      grid.info.origin.position.x = -HALF_W;
      grid.info.origin.position.y = -HALF_H;
      grid.info.origin.position.z = 0.0;
      grid.info.origin.orientation.w = 1.0;

      grid.data.assign(W * H, 0);

      const float Z_MIN = 0.2f, Z_MAX = 1.5f;
      for (const auto& lp : local_cones) {
        if (lp.z() < Z_MIN || lp.z() > Z_MAX) continue;
        float gx = lp.x() + HALF_W;
        float gy = lp.y() + HALF_H;
        int ix = static_cast<int>(std::floor(gx / RES));
        int iy = static_cast<int>(std::floor(gy / RES));
        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        int idx = iy * W + ix;
        grid.data[idx] = 100;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            int nx = ix + dx, ny = iy + dy;
            if (nx >= 0 && ny >= 0 && nx < W && ny < H)
              grid.data[ny * W + nx] = 100;
          }
        }
      }
      pub_grid_->publish(grid);
    }

    // --- 마지막 병합/퍼블리시 (한 번만) ---
    visualization_msgs::msg::MarkerArray merged;
    merged.markers.insert(merged.markers.end(),
                          cone_markers.markers.begin(), cone_markers.markers.end());
    merged.markers.insert(merged.markers.end(),
                          p12_arr.markers.begin(), p12_arr.markers.end());
    merged.markers.insert(merged.markers.end(),
                          group_markers.markers.begin(), group_markers.markers.end());

    pub_marker_->publish(merged);
    pub_marker_groups_->publish(group_markers);
  } // cloudCallback

  void publishMarkersOnly(const std_msgs::msg::Header& header,
                          const Eigen::Vector3f& p1,
                          std::optional<Eigen::Vector3f> p2_opt) {
    visualization_msgs::msg::MarkerArray arr;
    auto sphere = [&](int id, const Eigen::Vector3f& p,
                      float r, float g, float b, const std::string& ns){
      visualization_msgs::msg::Marker m;
      m.header = header;
      m.ns = ns;
      m.id = id;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = p.x();
      m.pose.position.y = p.y();
      m.pose.position.z = p.z();
      m.scale.x = m.scale.y = m.scale.z = 0.4;
      m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
      return m;
    };

    arr.markers.push_back(sphere(0, p1, 1.0, 0.0, 0.0, "closest_cone_1"));
    if (p2_opt.has_value()) {
      arr.markers.push_back(sphere(1, *p2_opt, 0.0, 1.0, 0.0, "closest_cone_2"));
    } else {
      // id=1 삭제
      visualization_msgs::msg::Marker del;
      del.header = header; del.ns="closest_cone_2"; del.id=1; del.action=visualization_msgs::msg::Marker::DELETE;
      arr.markers.push_back(del);
    }
    pub_marker_->publish(arr);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_groups_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_grid_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster_cloud_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_deg_zcloud_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingPreNode>());
  rclcpp::shutdown();
  return 0;
}