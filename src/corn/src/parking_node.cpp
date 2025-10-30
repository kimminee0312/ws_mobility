#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdio>   // for std::snprintf
#include <cstring>  // safety include for snprintf environments

using PointT = pcl::PointXYZ;

class ParkingNode : public rclcpp::Node {
public:
  ParkingNode()
  : Node("parking_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    prev_first_y_(std::numeric_limits<float>::quiet_NaN())
  {
    // ===== params =====
    target_frame_        = declare_parameter<std::string>("target_frame", "map");
    source_frame_        = declare_parameter<std::string>("source_frame", "lidar_link");
    base_frame_          = declare_parameter<std::string>("base_frame",   "lidar_link");

    window_size_         = declare_parameter<int>("centroid_window_size", 3000000);
    accum_tol_           = declare_parameter<double>("accum_cluster_tolerance", 0.20);
    accum_min_pts_       = declare_parameter<int>("accum_min_points", 3);
    accum_max_pts_       = declare_parameter<int>("accum_max_points", 10000);

    stable_merge_dist_   = declare_parameter<double>("stable_merge_dist", 0.30);
    stable_hits_needed_  = declare_parameter<int>("stable_confirm_hits", 3);
    stable_ttl_frames_   = declare_parameter<int>("stable_ttl_frames", 120);

    // ===== ROS I/O =====
    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/sensing/lidar/concatenated/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&ParkingNode::cloudCallback, this, std::placeholders::_1));

    pub_zcut_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/deg_points_deg_zfiltered_colored", 10);
    pub_cluster_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/deg_cone_clusters_colored", 10);
    pub_tf_debug_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/deg_test_tf_points", 10);

    pub_accum_centroids_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/deg_parking_pre/accum_centroids", 10);
    pub_stable_centroids_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/deg_parking_pre/stable_cones", 10);

    pub_all_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/parking_cone_markers", 10);

    pub_entrance_pose_ = create_publisher<geometry_msgs::msg::PoseArray>(
      "/cone_parking_entrance", 10);
    
    pub_goal_  = create_publisher<geometry_msgs::msg::PoseStamped>("/parking_goal", 10); 

    RCLCPP_INFO(get_logger(), "✅ ParkingNode ready.");
  }

private:
  // ===== tracking state =====
  struct Track {
    Eigen::Vector3f pos;
    int hits = 0;
    int last_seen = 0;
  };

  std::deque<std::vector<Eigen::Vector3f>> ring_map_centroids_;
  std::vector<Track> tracks_;
  int frame_counter_ = 0;

  float prev_first_y_;  // y-jump smoothing

  int window_size_;
  double accum_tol_;
  int accum_min_pts_;
  int accum_max_pts_;
  double stable_merge_dist_;
  int stable_hits_needed_;
  int stable_ttl_frames_;

  std::string target_frame_;
  std::string source_frame_;
  std::string base_frame_;

  // ===== ROS handles =====
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_zcut_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_tf_debug_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_accum_centroids_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_stable_centroids_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_all_markers_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_entrance_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_; 


  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // === entrance latch state ===
  bool entrance_locked_ = false;
  Eigen::Vector3f pA_locked_;
  Eigen::Vector3f pB_locked_;
  Eigen::Vector3f goal_locked_;
  geometry_msgs::msg::Quaternion goal_q_locked_;

  // ===== helpers =====
  static visualization_msgs::msg::MarkerArray makeSphereList(
      const std::vector<Eigen::Vector3f>& pts,
      const std_msgs::msg::Header& h,
      const std::string& ns,
      float r,float g,float b,float a=0.95f)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker m;
    m.header = h;
    m.ns = ns;
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = 0.25;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    for (auto &p : pts) {
      geometry_msgs::msg::Point q;
      q.x = p.x();
      q.y = p.y();
      q.z = 0.0;
      m.points.push_back(q);
    }
    if (m.points.empty()) {
      m.action = visualization_msgs::msg::Marker::DELETE;
    }
    arr.markers.push_back(m);
    return arr;
  }

  static sensor_msgs::msg::PointCloud2 makeCloudXYZ(
      const std::vector<Eigen::Vector3f>& pts,
      const std::string& frame_id,
      const rclcpp::Time& stamp)
  {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp    = stamp;
    cloud.height = 1;
    cloud.width  = static_cast<uint32_t>(pts.size());
    cloud.is_bigendian = false;
    cloud.is_dense     = true;

    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(cloud.width);

    sensor_msgs::PointCloud2Iterator<float> it_x(cloud,"x");
    sensor_msgs::PointCloud2Iterator<float> it_y(cloud,"y");
    sensor_msgs::PointCloud2Iterator<float> it_z(cloud,"z");

    for (auto &p : pts) {
      *it_x = p.x();
      *it_y = p.y();
      *it_z = p.z();
      ++it_x; ++it_y; ++it_z;
    }
    return cloud;
  }

  void accumulateFrame(const std::vector<Eigen::Vector3f>& frame_pts) {
    ring_map_centroids_.push_back(frame_pts);
    while ((int)ring_map_centroids_.size() > window_size_) {
      ring_map_centroids_.pop_front();
    }
  }

  std::vector<Eigen::Vector3f> flattenAccum() const {
    size_t total = 0;
    for (auto &v : ring_map_centroids_) total += v.size();
    std::vector<Eigen::Vector3f> out;
    out.reserve(total);
    for (auto &v : ring_map_centroids_) {
      out.insert(out.end(), v.begin(), v.end());
    }
    return out;
  }

  std::vector<Eigen::Vector3f> reclusterStable(const std::vector<Eigen::Vector3f>& pts) {
    if (pts.empty()) return {};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->points.reserve(pts.size());
    for (auto &p : pts) {
      cloud->points.emplace_back(p.x(), p.y(), p.z());
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(accum_tol_);
    ec.setMinClusterSize(accum_min_pts_);
    ec.setMaxClusterSize(accum_max_pts_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusters);

    std::vector<Eigen::Vector3f> out;
    out.reserve(clusters.size());
    for (auto &idx : clusters) {
      Eigen::Matrix<float,4,1> c4;
      pcl::compute3DCentroid(*cloud, idx, c4);
      out.emplace_back(c4.x(), c4.y(), c4.z());
    }
    return out;
  }

  std::vector<Eigen::Vector3f> updateTracks(const std::vector<Eigen::Vector3f>& stable_now) {
    ++frame_counter_;
    // match obs to tracks
    for (auto &z : stable_now) {
      int best = -1;
      double bd = 1e9;
      for (int i=0;i<(int)tracks_.size();++i) {
        double d = (tracks_[i].pos - z).norm();
        if (d < bd) { bd = d; best = i; }
      }
      if (best >= 0 && bd <= stable_merge_dist_) {
        tracks_[best].pos = 0.5f*(tracks_[best].pos + z);
        tracks_[best].hits += 1;
        tracks_[best].last_seen = frame_counter_;
      } else {
        Track t;
        t.pos = z;
        t.hits = 1;
        t.last_seen = frame_counter_;
        tracks_.push_back(t);
      }
    }
    // drop stale
    tracks_.erase(
      std::remove_if(
        tracks_.begin(), tracks_.end(),
        [&](const Track &t){
          return (frame_counter_ - t.last_seen) > stable_ttl_frames_;
        }),
      tracks_.end()
    );

    // confirmed tracks
    std::vector<Eigen::Vector3f> confirmed;
    for (auto &t : tracks_) {
      if (t.hits >= stable_hits_needed_) confirmed.push_back(t.pos);
    }
    return confirmed;
  }

  // ===== main callback =====
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1. raw cloud -> pcl
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    // 2. z pass-through
    pcl::PointCloud<PointT>::Ptr cloud_z(new pcl::PointCloud<PointT>);
    {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(-0.21f, 0.5f);
      pass.filter(*cloud_z);
    }
    if (cloud_z->empty()) return;

    // debug purple cloud
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr zcol(new pcl::PointCloud<pcl::PointXYZRGB>);
      zcol->points.reserve(cloud_z->points.size());
      for (auto &p : cloud_z->points) {
        pcl::PointXYZRGB pr;
        pr.x=p.x; pr.y=p.y; pr.z=p.z;
        pr.r=255; pr.g=0; pr.b=255;
        zcol->points.push_back(pr);
      }
      zcol->width  = (uint32_t)zcol->points.size();
      zcol->height = 1;
      zcol->is_dense=false;
      sensor_msgs::msg::PointCloud2 zmsg;
      pcl::toROSMsg(*zcol, zmsg);
      zmsg.header = msg->header;
      pub_zcut_cloud_->publish(zmsg);
    }

    // 3. first clustering
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

    std::vector<Eigen::Vector3f> centroids_sensor;
    pcl::PointCloud<PointT>::Ptr cloud_outer(new pcl::PointCloud<PointT>);

    for (auto &indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cls(new pcl::PointCloud<PointT>);
      cls->points.reserve(indices.indices.size());
      for (int idx : indices.indices) cls->push_back((*cloud_z)[idx]);

      Eigen::Vector4f min_pt, max_pt;
      pcl::getMinMax3D(*cls, min_pt, max_pt);
      float h = max_pt.z() - min_pt.z();
      float w = max_pt.x() - min_pt.x();
      float d = max_pt.y() - min_pt.y();

      bool cone_like =
        (h >= 0.1f && h <= 1.0f) &&
        (w <= 0.4f) &&
        (d <= 0.4f);
      if (!cone_like) {
        *cloud_outer += *cls;
        continue;
      }

      Eigen::Vector4f c4;
      pcl::compute3DCentroid(*cls, c4);
      centroids_sensor.push_back(c4.head<3>());
    }

    // 4. second clustering on leftovers
    if (!cloud_outer->empty()) {
      pcl::search::KdTree<PointT>::Ptr tree2(new pcl::search::KdTree<PointT>);
      tree2->setInputCloud(cloud_outer);

      std::vector<pcl::PointIndices> cluster_indices2;
      pcl::EuclideanClusterExtraction<PointT> ec2;
      ec2.setClusterTolerance(0.3);
      ec2.setMinClusterSize(2);
      ec2.setMaxClusterSize(1200);
      ec2.setSearchMethod(tree2);
      ec2.setInputCloud(cloud_outer);
      ec2.extract(cluster_indices2);

      static const std::array<std::array<uint8_t,3>,12> PALETTE = {{
        {255,59,48},{255,149,0},{255,204,0},{52,199,89},
        {0,199,190},{48,176,199},{88,86,214},{255,45,85},
        {142,142,147},{162,132,94},{64,156,255},{100,210,255}
      }};

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
      colored->is_dense=false;
      size_t cid=0;
      for (auto &idxs : cluster_indices2) {
        const auto &col = PALETTE[cid % PALETTE.size()];
        cid++;

        pcl::PointCloud<PointT>::Ptr cls2(new pcl::PointCloud<PointT>);
        cls2->points.reserve(idxs.indices.size());
        for (int i : idxs.indices) cls2->push_back((*cloud_outer)[i]);

        Eigen::Vector4f min2, max2;
        pcl::getMinMax3D(*cls2, min2, max2);
        float h2 = max2.z() - min2.z();
        float w2 = max2.x() - min2.x();
        float d2 = max2.y() - min2.y();
        bool cone_like2 =
          (h2 >= 0.1f && h2 <= 1.0f) &&
          (w2 <= 0.5f) &&
          (d2 <= 0.5f);
        if (!cone_like2) continue;

        Eigen::Vector4f c42;
        pcl::compute3DCentroid(*cls2, c42);
        centroids_sensor.push_back(c42.head<3>());

        for (auto &p : cls2->points) {
          pcl::PointXYZRGB pr;
          pr.x=p.x; pr.y=p.y; pr.z=p.z;
          pr.r=col[0]; pr.g=col[1]; pr.b=col[2];
          colored->points.push_back(pr);
        }
      }

      if (!colored->points.empty()) {
        sensor_msgs::msg::PointCloud2 outmsg;
        pcl::toROSMsg(*colored, outmsg);
        outmsg.header = msg->header;
        pub_cluster_cloud_->publish(outmsg);
      }
    }

    if (centroids_sensor.empty()) return;

    // 5. sensor -> map TF
    std::vector<Eigen::Vector3f> frame_centroids_map;
    sensor_msgs::msg::PointCloud2 cloud_sensor_xyz =
      makeCloudXYZ(centroids_sensor, source_frame_, msg->header.stamp);

    geometry_msgs::msg::TransformStamped tf_s2m;
    try {
      tf_s2m = tf_buffer_.lookupTransform(
        target_frame_,    // map
        source_frame_,    // lidar_link
        cloud_sensor_xyz.header.stamp
      );
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF sensor->map failed: %s", ex.what());
      return;
    }

    // debug transformed cloud
    {
      sensor_msgs::msg::PointCloud2 cloud_map_xyz;
      tf2::doTransform(cloud_sensor_xyz, cloud_map_xyz, tf_s2m);
      cloud_map_xyz.header.frame_id = target_frame_;
      pub_tf_debug_cloud_->publish(cloud_map_xyz);
    }

    {
      Eigen::Isometry3d T_map_from_src = tf2::transformToEigen(tf_s2m.transform);
      frame_centroids_map.reserve(centroids_sensor.size());
      for (auto &c : centroids_sensor) {
        Eigen::Vector3d v(c.x(), c.y(), c.z());
        Eigen::Vector3d vm = T_map_from_src * v;
        frame_centroids_map.emplace_back(
          (float)vm.x(), (float)vm.y(), (float)vm.z()
        );
      }
    }

    // 6. accumulate + recluster + track -> confirmed in map frame
    accumulateFrame(frame_centroids_map);
    auto accum_pts  = flattenAccum();
    auto stable_now = reclusterStable(accum_pts);
    auto confirmed  = updateTracks(stable_now);

    // publish accum + stable
    std_msgs::msg::Header map_header;
    map_header.frame_id = target_frame_;
    map_header.stamp    = msg->header.stamp;
    pub_accum_centroids_->publish(
      makeSphereList(accum_pts,  map_header, "accum_centroids", 0.2f,0.7f,1.0f));
    pub_stable_centroids_->publish(
      makeSphereList(confirmed, map_header, "stable_cones",    0.6f,0.2f,1.0f));

    // 7. transform confirmed cones map->lidar_link for local reasoning
    std::vector<Eigen::Vector3f> confirmed_bl;
    geometry_msgs::msg::TransformStamped tf_map2base;
    bool have_tf_map2base = true;
    try {
      tf_map2base = tf_buffer_.lookupTransform(
        base_frame_,    // lidar_link
        target_frame_,  // map
        msg->header.stamp
      );
    } catch (const tf2::TransformException &ex) {
      have_tf_map2base = false;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF map->lidar_link failed: %s", ex.what());
    }

    if (have_tf_map2base) {
      Eigen::Isometry3d T_bl_from_map = tf2::transformToEigen(tf_map2base.transform);
      confirmed_bl.reserve(confirmed.size());
      for (auto &c_map : confirmed) {
        Eigen::Vector3d p_map(c_map.x(), c_map.y(), c_map.z());
        Eigen::Vector3d p_bl = T_bl_from_map * p_map;
        confirmed_bl.emplace_back(
          (float)p_bl.x(), (float)p_bl.y(), (float)p_bl.z()
        );
      }
    } else {
      confirmed_bl = confirmed; // fallback
    }

    visualization_msgs::msg::MarkerArray markers_out;

    if (confirmed.empty()) {
      pub_all_markers_->publish(markers_out);
      return;
    }

    // draw all cones in blue (map frame)
    {
      int id=0;
      for (auto &c_map : confirmed) {
        visualization_msgs::msg::Marker m;
        m.header = map_header;
        m.ns = "all_cones";
        m.id = id++;
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = c_map.x();
        m.pose.position.y = c_map.y();
        m.pose.position.z = c_map.z();
        m.scale.x = 0.3;
        m.scale.y = 0.3;
        m.scale.z = 0.6;
        m.color.r = 0.0;
        m.color.g = 0.0;
        m.color.b = 1.0;
        m.color.a = 0.8;
        markers_out.markers.push_back(m);
      }
    }

    // ------ lane-like line building ------
    auto &centroids = confirmed_bl; // work in lidar_link

    // pick first_idx = nearest cone to ego
    int first_idx = -1;
    float min_dist = 1e9f;
    for (int i=0;i<(int)centroids.size();++i) {
      float d = std::hypot(centroids[i][0], centroids[i][1]);
      if (d < min_dist) {
        min_dist = d;
        first_idx = i;
      }
    }
    if (first_idx < 0) {
      pub_all_markers_->publish(markers_out);
      return;
    }

    // smooth jump in y
    if (!std::isnan(prev_first_y_)) {
      if (std::fabs(centroids[first_idx][1] - prev_first_y_) > 0.2f) {
        RCLCPP_WARN(this->get_logger(), "First point y jump detected, reverting to previous y");
        centroids[first_idx][1] = prev_first_y_;
      }
    }
    prev_first_y_ = centroids[first_idx][1];

    std::vector<int> selected;
    selected.push_back(first_idx);

    // visualize first point as red sphere (map frame coord!)
    {
      visualization_msgs::msg::Marker m;
      m.header = map_header;
      m.ns = "fitting_start";
      m.id = 0;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;

      Eigen::Vector3f p_map = confirmed[first_idx]; // same index, map frame
      m.pose.position.x = p_map.x();
      m.pose.position.y = p_map.y();
      m.pose.position.z = p_map.z();
      m.scale.x = m.scale.y = m.scale.z = 0.3;
      m.color.r = 1.0;
      m.color.g = 0.0;
      m.color.b = 0.0;
      m.color.a = 1.0;
      markers_out.markers.push_back(m);
    }

    // find second_idx in front of first_idx
    int second_idx = -1;
    float best_dist = 7.0f;
    float best_y_diff = 1e9f;
    float min_angle_rad = 20.0f * M_PI / 180.0f;

    for (int i=0;i<(int)centroids.size();++i) {
      if (i == first_idx) continue;
      float dx = centroids[i][0] - centroids[first_idx][0];
      float dy = centroids[i][1] - centroids[first_idx][1];
      float dz = centroids[i][2] - centroids[first_idx][2];
      float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
      float angle = std::atan2(dy, dx);

      if (dist <= 7.0f && std::fabs(dy) <= 0.2f) {
        if (dist < best_dist ||
            (std::fabs(dist - best_dist) < 1e-3f && std::fabs(dy) < best_y_diff))
        {
          if (std::fabs(angle) <= min_angle_rad) {
            best_dist = dist;
            best_y_diff = std::fabs(dy);
            min_angle_rad = angle;
            second_idx = i;
          }
        }
      }
    }

    if (second_idx < 0) {
      pub_all_markers_->publish(markers_out);
      return;
    }

    selected.push_back(second_idx);

    // local frame: x_axis points first->second
    Eigen::Vector3f p0_bl = centroids[first_idx];
    Eigen::Vector3f p1_bl = centroids[second_idx];
    Eigen::Vector3f x_axis = (p1_bl - p0_bl).normalized();
    Eigen::Vector3f z_axis(0,0,1);
    Eigen::Vector3f y_axis = z_axis.cross(x_axis).normalized();
    z_axis = x_axis.cross(y_axis).normalized();

    Eigen::Matrix3f R_local;
    R_local.col(0) = x_axis;
    R_local.col(1) = y_axis;
    R_local.col(2) = z_axis;

    // greedy extend forward
    int current = second_idx;
    while (true) {
      int next_idx = -1;
      float best_local_x = 7.0f;
      float best_y_diff2 = 1e9f;

      for (int i=0;i<(int)centroids.size();++i) {
        if (std::find(selected.begin(), selected.end(), i) != selected.end())
          continue;

        Eigen::Vector3f vec_bl = centroids[i] - centroids[current];
        Eigen::Vector3f local = R_local.transpose() * vec_bl;
        float lx = local[0];
        float ly = local[1];

        if (lx > 0.0f && lx <= 7.0f && std::fabs(ly) <= 0.15f) {
          if (lx < best_local_x ||
              (std::fabs(lx - best_local_x) < 1e-3f && std::fabs(ly) < best_y_diff2))
          {
            best_local_x = lx;
            best_y_diff2 = std::fabs(ly);
            next_idx = i;
          }
        }
      }

      if (next_idx == -1) break;
      selected.push_back(next_idx);
      current = next_idx;
    }

    // DEBUG: print line points
    {
      std::string s_bl = "[line pts BL] ";
      std::string s_map = "[line pts MAP] ";

      for (int idx : selected) {
        const auto &p_bl = confirmed_bl[idx];
        const auto &p_map = confirmed[idx];
        char buf1[128];
        char buf2[128];
        std::snprintf(buf1, sizeof(buf1), "(%.2f, %.2f, %.2f) ", p_bl.x(), p_bl.y(), p_bl.z());
        std::snprintf(buf2, sizeof(buf2), "(%.2f, %.2f, %.2f) ", p_map.x(), p_map.y(), p_map.z());
        s_bl  += buf1;
        s_map += buf2;
      }

      RCLCPP_INFO(this->get_logger(), "%s", s_bl.c_str());
      RCLCPP_INFO(this->get_logger(), "%s", s_map.c_str());
    }

    // draw fitted line (map frame)
    if (selected.size() >= 2) {
      visualization_msgs::msg::Marker line;
      line.header = map_header;
      line.ns = "fitting_line";
      line.id = 0;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.2;
      line.color.r = 0.53f;
      line.color.g = 0.81f;
      line.color.b = 0.92f;
      line.color.a = 0.3f;

      for (int idx : selected) {
        Eigen::Vector3f p_map = confirmed[idx]; // map coords
        geometry_msgs::msg::Point pt;
        pt.x = p_map.x();
        pt.y = p_map.y();
        pt.z = p_map.z();
        line.points.push_back(pt);
      }
      markers_out.markers.push_back(line);
    }

    // ===== ENTRANCE DETECTION + PARKING GOAL =====
    struct LocalPoint {
      int idx_global;
      float x_along;
      float y_side;
      Eigen::Vector3f p_map;
    };
    std::vector<LocalPoint> local_pts;
    local_pts.reserve(selected.size());

    // first_idx 기준 로컬 좌표 (R_local, p0_bl) 사용
    for (int idx : selected) {
      Eigen::Vector3f p_bl   = confirmed_bl[idx];
      Eigen::Vector3f rel_bl = p_bl - p0_bl;          // first point 기준
      Eigen::Vector3f p_loc  = R_local.transpose() * rel_bl;

      LocalPoint lp;
      lp.idx_global = idx;
      lp.x_along    = p_loc.x();
      lp.y_side     = p_loc.y();
      lp.p_map      = confirmed[idx];                 // map frame
      local_pts.push_back(lp);
    }

    // 전방 순서로 정렬
    std::sort(local_pts.begin(), local_pts.end(),
              [](const LocalPoint& A, const LocalPoint& B){
                return A.x_along < B.x_along;
              });

    if (!entrance_locked_){
      // 인접 쌍 중 gap_x 최대
      const float ENTRANCE_MIN_GAP = 4.0f; // 5m 이상
      float best_gap_x = -1.0f;
      int best_i = -1;
      int best_j = -1;

      for (size_t k=0; k+1<local_pts.size(); ++k) {
        float gap_x = std::fabs(local_pts[k+1].x_along - local_pts[k].x_along);
        if (gap_x > best_gap_x) {
          best_gap_x = gap_x;
          best_i = (int)k;
          best_j = (int)(k+1);
        }
      }

      // 인접 쌍 중 gap_x 최대 ...
      if (best_i != -1 && best_j != -1 && best_gap_x >= ENTRANCE_MIN_GAP) {
        Eigen::Vector3f pA = local_pts[best_i].p_map;
        Eigen::Vector3f pB = local_pts[best_j].p_map;

        // --- parking_goal 위치 ---
        Eigen::Vector3f mid;
        mid.x() = 0.5f * (pA.x() + pB.x());
        mid.y() = 0.5f * (pA.y() + pB.y());
        mid.z() = 0.5f * (pA.z() + pB.z());

        Eigen::Vector3f goal_map;
        goal_map.x() = mid.x();
        goal_map.y() = mid.y() + 1.3f;   // 요구사항: avg_y + 2.5
        goal_map.z() = mid.z();

        // --- parking_goal 방향: 진행방향(차선 따라 전진하는 방향) ---
        // 1) lidar_link -> map 회전 추출
        Eigen::Isometry3d T_map_from_base =
          tf2::transformToEigen(tf_map2base.transform).inverse();

        // 2) x_axis (lidar_link 진행방향) 를 map frame으로 회전만 적용
        Eigen::Vector3d x_axis_bl_dir(x_axis.x(), x_axis.y(), x_axis.z());
        Eigen::Vector3d x_axis_map_dir =
          T_map_from_base.rotation() * x_axis_bl_dir;

        // 3) yaw in map
        float yaw_goal = std::atan2(
          (float)x_axis_map_dir.y(),
          (float)x_axis_map_dir.x()
        );

        // yaw -> quaternion (roll=0,pitch=0)
        geometry_msgs::msg::Quaternion q_goal;
        {
          double cy = std::cos(yaw_goal * 0.5);
          double sy = std::sin(yaw_goal * 0.5);
          double cr = 1.0;
          double sr = 0.0;
          double cp = 1.0;
          double sp = 0.0;

          q_goal.w = cy*cr*cp + sy*sr*sp;
          q_goal.x = cy*sr*cp - sy*cr*sp;
          q_goal.y = cy*cr*sp + sy*sr*cp;
          q_goal.z = sy*cr*cp - cy*sr*sp;
        }

        // state keep
        entrance_locked_ = true;
        pA_locked_       = pA;
        pB_locked_       = pB;
        goal_locked_     = goal_map;
        goal_q_locked_   = q_goal;

        RCLCPP_INFO(this->get_logger(),
          "Entrance LOCKED: gap_x=%.2f m >= %.2f\n"
          " goal=(%.2f, %.2f, %.2f)",
          best_gap_x, ENTRANCE_MIN_GAP,
          goal_map.x(), goal_map.y(), goal_map.z()
        );
      } else {
        RCLCPP_WARN(this->get_logger(),
          "[entrance] none to lock this frame");
      }
    }

    if (entrance_locked_){
      // --- entrance cones (빨간 실린더) ---
      visualization_msgs::msg::Marker cyl1;
      cyl1.header = map_header;
      cyl1.ns = "entrance_points";
      cyl1.id = 2001;
      cyl1.type = visualization_msgs::msg::Marker::CYLINDER;
      cyl1.action = visualization_msgs::msg::Marker::ADD;
      cyl1.pose.position.x = pA_locked_.x();
      cyl1.pose.position.y = pA_locked_.y();
      cyl1.pose.position.z = pA_locked_.z();
      cyl1.scale.x = 0.4;
      cyl1.scale.y = 0.4;
      cyl1.scale.z = 0.6;
      cyl1.color.r = 1.0;
      cyl1.color.g = 0.0;
      cyl1.color.b = 0.0;
      cyl1.color.a = 1.0;
      markers_out.markers.push_back(cyl1);

      visualization_msgs::msg::Marker cyl2 = cyl1;
      cyl2.id = 2002;
      cyl2.pose.position.x = pB_locked_.x();
      cyl2.pose.position.y = pB_locked_.y();
      cyl2.pose.position.z = pB_locked_.z();
      markers_out.markers.push_back(cyl2);

      // RViz marker: parking_goal 화살표 (초록)
      visualization_msgs::msg::Marker goal_marker;
      goal_marker.header = map_header;
      goal_marker.ns = "parking_goal";
      goal_marker.id = 3001;
      goal_marker.type = visualization_msgs::msg::Marker::ARROW;
      goal_marker.action = visualization_msgs::msg::Marker::ADD;
      goal_marker.pose.position.x = goal_locked_.x();
      goal_marker.pose.position.y = goal_locked_.y();
      goal_marker.pose.position.z = goal_locked_.z();
      goal_marker.pose.orientation = goal_q_locked_;
      goal_marker.scale.x = 2.0;
      goal_marker.scale.y = 0.4;
      goal_marker.scale.z = 0.4;
      goal_marker.color.r = 0.0;
      goal_marker.color.g = 1.0;
      goal_marker.color.b = 0.0;
      goal_marker.color.a = 1.0;
      markers_out.markers.push_back(goal_marker);

      // PoseArray (/cone_parking_entrance): A,B,goal
      {
        geometry_msgs::msg::PoseArray pa_msg;
        pa_msg.header = map_header;

        geometry_msgs::msg::Pose poseA, poseB, poseGoal;
        poseA.position.x = pA_locked_.x();
        poseA.position.y = pA_locked_.y();
        poseA.position.z = pA_locked_.z();
        poseB.position.x = pB_locked_.x();
        poseB.position.y = pB_locked_.y();
        poseB.position.z = pB_locked_.z();

        poseGoal.position.x = goal_locked_.x();
        poseGoal.position.y = goal_locked_.y();
        poseGoal.position.z = goal_locked_.z();
        poseGoal.orientation = goal_q_locked_;

        pa_msg.poses.push_back(poseA);
        pa_msg.poses.push_back(poseB);
        pa_msg.poses.push_back(poseGoal);
        pub_entrance_pose_->publish(pa_msg);
      }
      // === (B) /parking_goal 단일 PoseStamped publish ===
      {
        geometry_msgs::msg::PoseStamped goal_msg;
        goal_msg.header = map_header;
        goal_msg.pose.position.x = goal_locked_.x();
        goal_msg.pose.position.y = goal_locked_.y();
        goal_msg.pose.position.z = goal_locked_.z();
        goal_msg.pose.orientation = goal_q_locked_;
        pub_goal_->publish(goal_msg);
      }
    } else {
      // 아직 락 안 됐으면 그냥 publish 안 하거나, 디버그만 하고 넘어가
      RCLCPP_DEBUG(this->get_logger(), "Entrance not locked yet => no goal publish this frame");
    }
  
    pub_all_markers_->publish(markers_out);
  } // end cloudCallback
}; // class ParkingNode


int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingNode>());
  rclcpp::shutdown();
  return 0;
}