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

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    has_published_tf_ = false;
    RCLCPP_INFO(this->get_logger(), "✅ ParkingPreNode started.");
  }

private:
  bool has_published_tf_;
  bool has_initialized_frame_ = false;
  Eigen::Vector3f origin_;
  Eigen::Matrix3f R_;

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // --- 이전 마커 전체 삭제 (잔상 제거) ---
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.push_back(clear_marker);
    pub_marker_->publish(clear_markers);

    // PCL 변환
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    // 클러스터링
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);   // 필요 시 조정
    ec.setMinClusterSize(3);
    ec.setMaxClusterSize(100);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<Eigen::Vector3f> centroids;
    centroids.reserve(cluster_indices.size());

    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
      cluster->points.reserve(indices.indices.size());
      for (int idx : indices.indices) cluster->push_back((*cloud)[idx]);

      // 라바콘 형태 필터 (AABB)
      Eigen::Vector4f min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);
      float height = max_pt.z() - min_pt.z();
      float width  = max_pt.x() - min_pt.x();
      float depth  = max_pt.y() - min_pt.y();
      if (height < 0.15f || height > 1.0f) continue;
      if (width  > 0.4f || depth  > 0.4f) continue;

      // 중심점
      Eigen::Vector4f c;
      pcl::compute3DCentroid(*cluster, c);
      centroids.push_back(c.head<3>());
    }

    if (centroids.size() < 1) return;

    // ✅ 모든 클러스터(라바콘 후보)를 파란색 원기둥으로 시각화
    visualization_msgs::msg::MarkerArray cone_markers;
    int marker_id = 0;
    for (const auto& c : centroids) {
      visualization_msgs::msg::Marker m;
      m.header = msg->header;
      m.ns = "all_cones";
      m.id = marker_id++;
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
      m.color.b = 1.0;   // 🔵 파란색
      m.color.a = 0.8;
      cone_markers.markers.push_back(m);
    }

    // 차량 원점(0,0,0) 기준 1번(가장 가까운) 찾기
    auto dist2d = [](const Eigen::Vector3f& p){ return std::hypot(p.x(), p.y()); };
    int first_idx = -1;
    float best = 1e9f;
    for (int i = 0; i < (int)centroids.size(); ++i) {
      float d = dist2d(centroids[i]);
      if (d < best) { best = d; first_idx = i; }
    }
    if (first_idx < 0) return;

    // 2번(두 번째로 가까운) 찾기: 차량 원점 기준으로 가깝되, p1과의 수평거리 ≤ 1.5 m 조건 추가
    const Eigen::Vector3f p1 = centroids[first_idx];

    int second_idx = -1;
    best = 1e9f;

    const float MAX_P12_DIST = 1.5f;  // p1과 p2 사이 최대 허용 수평거리 (미터)

    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
      if (i == first_idx) continue;

      // 차량 원점으로부터의 2D 거리(기존 우선순위 기준)
      float d_vehicle = std::hypot(centroids[i].x(), centroids[i].y());

      // p1과의 2D 수평거리(조건)
      float d12_xy = std::hypot(centroids[i].x() - p1.x(),
                                centroids[i].y() - p1.y());

      // 조건: p1과 1.0m 이내인 후보들 중에서 차량 원점 기준 가장 가까운 것을 선택
      if (d12_xy <= MAX_P12_DIST && d_vehicle < best) {
        best = d_vehicle;
        second_idx = i;
      }
    }

    if (second_idx < 0) {
      RCLCPP_WARN(this->get_logger(),
                  "No valid second point within %.2fm from p1. Publish p1 only.",
                  MAX_P12_DIST);
      publishMarkersOnly(msg->header, centroids[first_idx], std::nullopt);
      return;
    }
    const Eigen::Vector3f p2 = centroids[second_idx];

    // ─────────────────────────────────────────────
    // 1) 마커: 1번(빨간 구), 2번(빨간 구), 1→2 화살표(빨강)
    // ─────────────────────────────────────────────
    visualization_msgs::msg::MarkerArray arr;

    auto sphere = [&](int id, const Eigen::Vector3f& p,
                      float r, float g, float b, const std::string& ns){
      visualization_msgs::msg::Marker m;
      m.header = msg->header;
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

    auto arrow = [&](int id, const Eigen::Vector3f& a, const Eigen::Vector3f& b){
      visualization_msgs::msg::Marker m;
      m.header = msg->header;
      m.ns = "arrow_1_to_2";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::ARROW;
      m.action = visualization_msgs::msg::Marker::ADD;
      geometry_msgs::msg::Point pA, pB;
      pA.x = a.x(); pA.y = a.y(); pA.z = a.z();
      pB.x = b.x(); pB.y = b.y(); pB.z = b.z();
      m.points.push_back(pA);
      m.points.push_back(pB);
      m.scale.x = 0.10;  // shaft diameter
      m.scale.y = 0.20;  // head diameter
      m.scale.z = 0.20;  // head length
      m.color.r = 1.0f; 
      m.color.g = 0.0f; 
      m.color.b = 0.0f; 
      m.color.a = 1.0f;
      return m;
    };

    arr.markers.push_back(sphere(0, p1, 1.0, 0.0, 0.0, "closest_cone_1")); // 빨강
    arr.markers.push_back(sphere(1, p2, 1.0, 0.0, 0.0, "closest_cone_2")); // 빨강
    arr.markers.push_back(arrow(2, p1, p2));                               // 화살표
    pub_marker_->publish(arr);

    // ─────────────────────────────────────────────
    // 2) 로컬 좌표계 정의 (원점 = p1, x+: p1→p2 방향, y+: 오른쪽, z+: 위)
    //    - z_global = (0,0,1)
    //    - x_axis = normalize(p2 - p1)
    //    - y_axis = x_axis × z_global  (오른쪽이 되도록)
    //    - z_axis = x_axis × y_axis    (정규직교 보정)
    // ─────────────────────────────────────────────
    Eigen::Vector3f x_axis = (p2 - p1);
    float len = x_axis.norm();
    if (len < 1e-6f) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "p1 and p2 are too close; skip TF.");
      return;
    }
    x_axis /= len;

    const Eigen::Vector3f z_global(0.f, 0.f, 1.f);
    Eigen::Vector3f y_axis = x_axis.cross(z_global);  // 오른쪽
    if (y_axis.norm() < 1e-6f) {
      // x가 z와 거의 평행하면 y를 임의로 설정
      y_axis = Eigen::Vector3f(0.f, 1.f, 0.f);
    } else {
      y_axis.normalize();
    }
    Eigen::Vector3f z_axis = x_axis.cross(y_axis);
    z_axis.normalize();

    // 회전 행렬 (열에 축 벡터)
    Eigen::Matrix3f R;
    R.col(0) = x_axis;
    R.col(1) = y_axis;
    R.col(2) = z_axis;

    // 행렬 → 쿼터니언
    Eigen::Quaternionf q(R);
    q.normalize();

    // ─────────────────────────────────────────────
    // 3) TF 브로드캐스트
    //    parent: msg->header.frame_id (센서 프레임)
    //    child : "cone_local"
    // ─────────────────────────────────────────────
    geometry_msgs::msg::TransformStamped T;
    T.header = msg->header;  // stamp & parent frame
    T.child_frame_id = "cone_local";
    T.transform.translation.x = p1.x();
    T.transform.translation.y = p1.y();
    T.transform.translation.z = p1.z();
    T.transform.rotation.x = q.x();
    T.transform.rotation.y = q.y();
    T.transform.rotation.z = q.z();
    T.transform.rotation.w = q.w();

    // ─────────────────────────────────────────────
    // TF 브로드캐스트 (한 번만)
    // ─────────────────────────────────────────────
    if (!has_published_tf_) {
      tf_broadcaster_->sendTransform(T);
      has_published_tf_ = true;
      RCLCPP_INFO(this->get_logger(),
                  "==================== published cone_local TF once (frame=%s) ====================",
                   msg->header.frame_id.c_str());
    }

    // 로그
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "p1=(%.2f,%.2f,%.2f), p2=(%.2f,%.2f,%.2f), dist=%.2fm. TF frame: %s->cone_local",
      p1.x(), p1.y(), p1.z(), p2.x(), p2.y(), p2.z(), (p2 - p1).norm(),
      msg->header.frame_id.c_str());

    // --- 좌표계 한 번만 정의 ---
    if (!has_initialized_frame_) {
      origin_ = p1;
      R_ = R;  // 회전행렬 저장
      has_initialized_frame_ = true;
      RCLCPP_INFO(this->get_logger(), "✅ Local frame initialized (cone_local)");
    }

    // --- 1️⃣ 이후부터는 cone_local 기준으로 변환 ---
    std::vector<Eigen::Vector3f> local_cones;
    local_cones.reserve(centroids.size());
    for (const auto& c : centroids) {
      Eigen::Vector3f local = R_.transpose() * (c - origin_);
      local_cones.push_back(local);
    }

    // --- 그룹화 (X좌표 기준) ---
    std::vector<std::vector<Eigen::Vector3f>> groups_x;
    {
      std::vector<Eigen::Vector3f> sorted = local_cones;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.x() < b.x(); });

      std::vector<Eigen::Vector3f> current_group;
      for (size_t i = 0; i < sorted.size(); ++i) {
        if (current_group.empty()) {
          current_group.push_back(sorted[i]);
        } else {
          float diff_x  = std::fabs(sorted[i].x() - current_group.back().x());
          float dist_xy = (sorted[i] - current_group.back()).norm(); // 3D 거리
          // 같은 세로줄 + 인접거리 1.5m 이하
          if (diff_x < 0.1f && dist_xy < 1.5f) {
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
    {
      std::vector<Eigen::Vector3f> sorted = local_cones;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.y() < b.y(); });

      std::vector<Eigen::Vector3f> current_group;
      for (size_t i = 0; i < sorted.size(); ++i) {
        if (current_group.empty()) {
          current_group.push_back(sorted[i]);
        } else {
          float diff_y  = std::fabs(sorted[i].y() - current_group.back().y());
          float dist_xy = (sorted[i] - current_group.back()).norm(); // 3D 거리
          // 같은 가로줄 + 인접거리 1.5m 이하
          if (diff_y < 0.1f && dist_xy < 1.5f) {
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
  

    // --- 그룹화 결과 출력 ---
    RCLCPP_INFO(this->get_logger(), "📌 그룹화 결과 (cone_local 좌표계 기준)");

    RCLCPP_INFO(this->get_logger(), "X축 기준 그룹 수: %zu", groups_x.size());
    for (size_t g = 0; g < groups_x.size(); ++g) {
      std::ostringstream oss;
      oss << "  Group X" << g << ": ";
      for (const auto& p : groups_x[g]) {
        oss << "(" << std::fixed << std::setprecision(2)
            << p.x() << "," << p.y() << "," << p.z() << ") ";
      }
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    RCLCPP_INFO(this->get_logger(), "Y축 기준 그룹 수: %zu", groups_y.size());
    for (size_t g = 0; g < groups_y.size(); ++g) {
      std::ostringstream oss;
      oss << "  Group Y" << g << ": ";
      for (const auto& p : groups_y[g]) {
        oss << "(" << std::fixed << std::setprecision(2)
            << p.x() << "," << p.y() << "," << p.z() << ") ";
      }
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    // --- 3️⃣ 시각화: 그룹을 선(라인)으로 표시 ---
    visualization_msgs::msg::MarkerArray group_markers;
    std::vector<std::array<float,3>> colors_x = {
      {1.0f,0.0f,0.0f}, {1.0f,0.5f,0.0f}, {1.0f,0.0f,0.5f}, {0.8f,0.3f,0.3f}
    };
    std::vector<std::array<float,3>> colors_y = {
      {0.0f,0.0f,1.0f}, {0.0f,1.0f,1.0f}, {0.3f,0.3f,0.8f}, {0.0f,0.5f,1.0f}
    };

    // 🔹 X기준 그룹: LINE_STRIP
    for (size_t g = 0; g < groups_x.size(); ++g) {
      auto color = colors_x[g % colors_x.size()];
      visualization_msgs::msg::Marker line;
      line.header = msg->header;
      line.ns = "group_x_line_" + std::to_string(g);
      line.id = 0;  // 그룹당 1개 라인
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.05;  // 선 두께
      line.color.r = color[0];
      line.color.g = color[1];
      line.color.b = color[2];
      line.color.a = 0.95f;

      // 로컬 -> 글로벌 변환하여 line.points에 push
      for (const auto& p_local : groups_x[g]) {
        Eigen::Vector3f global = origin_ + R_ * p_local;
        geometry_msgs::msg::Point P;
        P.x = global.x(); P.y = global.y(); P.z = global.z();
        line.points.push_back(P);
      }

      if (line.points.size() >= 2) {
        group_markers.markers.push_back(line);
      } 
    }

    // 🔹 Y기준 그룹: LINE_STRIP
    for (size_t g = 0; g < groups_y.size(); ++g) {
      auto color = colors_y[g % colors_y.size()];
      visualization_msgs::msg::Marker line;
      line.header = msg->header;
      line.ns = "group_y_line_" + std::to_string(g);
      line.id = 0;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.05;
      line.color.r = color[0];
      line.color.g = color[1];
      line.color.b = color[2];
      line.color.a = 0.95f;

      for (const auto& p_local : groups_y[g]) {
        Eigen::Vector3f global = origin_ + R_ * p_local;
        geometry_msgs::msg::Point P;
        P.x = global.x(); P.y = global.y(); P.z = global.z();
        line.points.push_back(P);
      }

      if (line.points.size() >= 2) {
        group_markers.markers.push_back(line);
      } 
    }

     // --- OccupancyGrid 생성/퍼블리시 (탑다운 2D) ---
    {
      const float RES = 0.10f;     // 10cm 해상도
      const int   W   = 200;       // 20m 폭
      const int   H   = 200;       // 20m 높이
      const float HALF_W = (W * RES) * 0.5f;
      const float HALF_H = (H * RES) * 0.5f;

      nav_msgs::msg::OccupancyGrid grid;
      grid.header = msg->header;
      grid.header.frame_id = "cone_local";    // 탑다운 기준 프레임
      grid.info.resolution = RES;
      grid.info.width  = W;
      grid.info.height = H;

      // cone_local 좌표계 기준 중앙정렬: 원점을 (-HALF_W, -HALF_H)에 둔다.
      grid.info.origin.position.x = -HALF_W;
      grid.info.origin.position.y = -HALF_H;
      grid.info.origin.position.z = 0.0;
      grid.info.origin.orientation.w = 1.0;   // 단위 회전

      grid.data.assign(W * H, 0);            // 0=free, 100=occupied, -1=unknown

      const float Z_MIN = -0.2f, Z_MAX = 1.5f; // 높이 필터(선택)
      for (const auto& lp : local_cones) {
        if (lp.z() < Z_MIN || lp.z() > Z_MAX) continue;

        // lp는 cone_local 좌표. (x,y)를 그리드 index로 투영
        float gx = lp.x() + HALF_W;
        float gy = lp.y() + HALF_H;
        int ix = static_cast<int>(std::floor(gx / RES));
        int iy = static_cast<int>(std::floor(gy / RES));
        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        int idx = iy * W + ix;

        grid.data[idx] = 100; // 점유

        // (선택) 주변 8방향 1셀 팽창
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
    
    // --- 마지막 병합/퍼블리시 ---
    visualization_msgs::msg::MarkerArray merged;
    // 1) 모든 라바콘 후보(파란 원기둥) 유지
    merged.markers.insert(merged.markers.end(),
                          cone_markers.markers.begin(), cone_markers.markers.end());
    // 2) 기준점/화살표 유지
    merged.markers.insert(merged.markers.end(),
                          arr.markers.begin(), arr.markers.end());
    // 3) 그룹 선 시각화 추가
    merged.markers.insert(merged.markers.end(),
                          group_markers.markers.begin(), group_markers.markers.end());

    // 퍼블리시
    pub_marker_->publish(merged);
    pub_marker_groups_->publish(group_markers);
    } // <-- 반드시 cloudCallback 함수 닫는 중괄호


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
    }

    // ✅ 실제 퍼블리시 추가
    pub_marker_->publish(arr);
  }
  
  

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_groups_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_grid_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingPreNode>());
  rclcpp::shutdown();
  return 0;
}
