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
    ec.setClusterTolerance(0.4);   // 필요 시 조정
    ec.setMinClusterSize(3);
    ec.setMaxClusterSize(60);
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
      if (height < 0.2f || height > 1.0f) continue;
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

    // 2번(두 번째로 가까운) 찾기 (차량 원점 기준, 1번 제외)
    int second_idx = -1;
    best = 1e9f;
    for (int i = 0; i < (int)centroids.size(); ++i) {
      if (i == first_idx) continue;
      float d = dist2d(centroids[i]);
      if (d < best) { best = d; second_idx = i; }
    }
    if (second_idx < 0) {
      // 1개만 있을 때는 1번만 표시하고 종료
      publishMarkersOnly(msg->header, centroids[first_idx], std::nullopt);
      return;
    }

    const Eigen::Vector3f p1 = centroids[first_idx];
    const Eigen::Vector3f p2 = centroids[second_idx];

    // ─────────────────────────────────────────────
    // 1) 마커: 1번(빨간 구), 2번(초록 구), 1→2 화살표(파랑)
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
                "==================== published cone_local TF once (frame=%s) ====================", msg->header.frame_id.c_str());
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
    for (const auto& c : centroids) {
    Eigen::Vector3f local = R_.transpose() * (c - origin_);
    local_cones.push_back(local);
    }

    // --- 2️⃣ 그룹화 수행 (x축, y축 별도) ---
    std::vector<std::vector<Eigen::Vector3f>> groups_x;
    std::vector<std::vector<Eigen::Vector3f>> groups_y;
    std::vector<bool> used_x(local_cones.size(), false);
    std::vector<bool> used_y(local_cones.size(), false);

    // 🔹 X좌표 기준 그룹화
    for (const auto& pt : local_cones) {
        bool assigned = false;
        for (auto& group : groups_x) {
            // 그룹 평균 x 계산
            float avg_x = 0.0f;
            for (auto& gpt : group) avg_x += gpt.x();
            avg_x /= group.size();

            if (fabs(pt.x() - avg_x) < 0.25f) {   // ← 🔧 완화된 조건
                group.push_back(pt);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            groups_x.push_back({pt});
        }
    }

    // 🔹 Y좌표 기준 그룹화
    for (const auto& pt : local_cones) {
        bool assigned = false;
        for (auto& group : groups_y) {
            // 그룹 평균 y 계산
            float avg_y = 0.0f;
            for (auto& gpt : group) avg_y += gpt.y();
            avg_y /= group.size();

            if (fabs(pt.y() - avg_y) < 0.15f) {   // ← 🔧 완화된 조건
                group.push_back(pt);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            groups_y.push_back({pt});
        }
    }


    // --- 3️⃣ 시각화 ---
    visualization_msgs::msg::MarkerArray group_markers;
    int id = 0;
    std::vector<std::array<float,3>> colors_x = {
    {1.0,0.0,0.0}, {1.0,0.5,0.0}, {1.0,0.0,0.5}, {0.8,0.3,0.3}
    };
    std::vector<std::array<float,3>> colors_y = {
    {0.0,0.0,1.0}, {0.0,1.0,1.0}, {0.3,0.3,0.8}, {0.0,0.5,1.0}
    };

    // 🔹 X기준 그룹 (세로줄)
    for (size_t g = 0; g < groups_x.size(); ++g) {
    auto color = colors_x[g % colors_x.size()];
    for (size_t i = 0; i < groups_x[g].size(); ++i) {
        Eigen::Vector3f global = origin_ + R_ * groups_x[g][i];
        visualization_msgs::msg::Marker m;
        m.header = msg->header;
        m.ns = "group_x_" + std::to_string(g);  // ✅ 그룹마다 namespace 구분
        m.id = static_cast<int>(i);             // ✅ 그룹 내 개별 id
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = global.x();
        m.pose.position.y = global.y();
        m.pose.position.z = global.z();
        m.scale.x = m.scale.y = 0.35;
        m.scale.z = 0.7;
        m.color.r = color[0];
        m.color.g = color[1];
        m.color.b = color[2];
        m.color.a = 0.3;
        group_markers.markers.push_back(m);
    }
    }

    // 🔹 Y기준 그룹 (가로줄)
    for (size_t g = 0; g < groups_y.size(); ++g) {
    auto color = colors_y[g % colors_y.size()];
    for (size_t i = 0; i < groups_y[g].size(); ++i) {
        Eigen::Vector3f global = origin_ + R_ * groups_y[g][i];
        visualization_msgs::msg::Marker m;
        m.header = msg->header;
        m.ns = "group_y_" + std::to_string(g);  // ✅ 그룹마다 namespace 구분
        m.id = static_cast<int>(i);             // ✅ 그룹 내 개별 id
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = global.x();
        m.pose.position.y = global.y();
        m.pose.position.z = global.z();
        m.scale.x = m.scale.y = 0.35;
        m.scale.z = 0.7;
        m.color.r = color[0];
        m.color.g = color[1];
        m.color.b = color[2];
        m.color.a = 0.3;
        group_markers.markers.push_back(m);
    }
    }

    // --- 마지막 부분 ---
    visualization_msgs::msg::MarkerArray merged;

    // 1️⃣ 파란색 라바콘
    merged.markers.insert(merged.markers.end(),
                        cone_markers.markers.begin(), cone_markers.markers.end());
    // 2️⃣ 두 기준점과 화살표
    merged.markers.insert(merged.markers.end(),
                        arr.markers.begin(), arr.markers.end());
    // 3️⃣ 그룹 시각화
    merged.markers.insert(merged.markers.end(),
                        group_markers.markers.begin(), group_markers.markers.end());

    // 🔹 한 번만 퍼블리시
    pub_marker_->publish(merged);

    // 🔹 그룹 마커만 별도 토픽으로도 발행
    pub_marker_groups_->publish(group_markers);
  }

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
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingPreNode>());
  rclcpp::shutdown();
  return 0;
}
