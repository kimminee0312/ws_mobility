#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

using MarkerArray = visualization_msgs::msg::MarkerArray;
using Marker      = visualization_msgs::msg::Marker;
using Point       = geometry_msgs::msg::Point;
using std::placeholders::_1;

// ---------- [사각형 핏팅 유틸: 단일 파일 내에 정의] ----------
static bool fit_min_area_rect(const std::vector<cv::Point2f>& in_pts,
                              std::array<cv::Point2f,4>& box_out)
{
  if (in_pts.size() < 3) return false;

  // 유효점만 사용
  std::vector<cv::Point2f> pts; pts.reserve(in_pts.size());
  for (auto& p : in_pts) {
    if (std::isfinite(p.x) && std::isfinite(p.y)) pts.push_back(p);
  }
  if (pts.size() < 3) return false;

  // Convex Hull
  std::vector<cv::Point2f> hull;
  cv::convexHull(pts, hull, /*clockwise=*/true, /*returnPoints=*/true);
  if (hull.size() < 3) return false;

  // 최소면적 외접사각형
  cv::RotatedRect rr = cv::minAreaRect(hull);
  cv::Point2f box[4]; rr.points(box);
  for (int i=0;i<4;++i) box_out[i] = box[i];
  return true;
}

// 폭/길이 규격 보정(옵션) : 주차칸 규격 prior를 허용오차 내로 클램프
static void clamp_rect_wh(std::array<cv::Point2f,4>& box,
                          float w_nom, float h_nom, float w_tol, float h_tol)
{
  auto L=[](cv::Point2f a, cv::Point2f b){ return std::hypot(b.x-a.x, b.y-a.y); };
  float e0 = L(box[0], box[1]);
  float e1 = L(box[1], box[2]);
  float w  = std::min(e0, e1);
  float h  = std::max(e0, e1);

  float w_target = std::clamp(w, w_nom - w_tol, w_nom + w_tol);
  float h_target = std::clamp(h, h_nom - h_tol, h_nom + h_tol);

  // 중심과 축
  cv::Point2f c(0,0); for (auto& q: box) { c += q; } c.x/=4.f; c.y/=4.f;
  cv::Point2f u = box[1] - box[0];
  float Lu = std::max(1e-6f, std::hypot(u.x, u.y));
  u *= (1.f / Lu);
  cv::Point2f v(-u.y, u.x);

  // e0가 긴 변이면 w/h 스왑
  if (e0 > e1) std::swap(w_target, h_target);

  // 재구성(시계방향)
  std::array<cv::Point2f,4> b;
  cv::Point2f du = 0.5f * w_target * u;
  cv::Point2f dv = 0.5f * h_target * v;
  b[0] = c - du - dv;
  b[1] = c + du - dv;
  b[2] = c + du + dv;
  b[3] = c - du + dv;
  box = b;
}
// ------------------------------------------------------------

class RectFitterNode : public rclcpp::Node {
public:
  RectFitterNode() : Node("rect_fitter_node")
  {
    frame_id_ = declare_parameter<std::string>("frame_id", "velodyne");
    in_topic_ = declare_parameter<std::string>("in_topic", "/parking_group_markers");
    w_nom_    = declare_parameter("slot_width_nom",   3.4); // m
    h_nom_    = declare_parameter("slot_length_nom",  5.0); // m
    w_tol_    = declare_parameter("slot_width_tol",   0.6); // m
    h_tol_    = declare_parameter("slot_length_tol",  0.9); // m
    line_width_ = declare_parameter("line_width", 0.05);

    sub_ = create_subscription<MarkerArray>(in_topic_, 1, std::bind(&RectFitterNode::cb, this, _1));
    pub_ = create_publisher<MarkerArray>("/slot_rects", 1);
  }

private:
  void cb(const MarkerArray::SharedPtr msg)
  {
    MarkerArray out;
    rclcpp::Time now = this->now();
    int id_line = 0, id_text = 0;

    for (const auto& mk : msg->markers) {
      // 그룹 입력은 POINTS 또는 LINE_STRIP 아무거나 허용
      if (mk.type != Marker::POINTS && mk.type != Marker::LINE_STRIP) continue;
      if (mk.points.size() < 3) continue;

      std::vector<cv::Point2f> pts; pts.reserve(mk.points.size());
      for (const auto& p : mk.points) pts.emplace_back(p.x, p.y);

      std::array<cv::Point2f,4> box;
      if (!fit_min_area_rect(pts, box)) continue;

      // 규격 보정(필요 없으면 주석 처리)
      clamp_rect_wh(box, (float)w_nom_, (float)h_nom_, (float)w_tol_, (float)h_tol_);

      // LINE_LIST로 사각형 출력
      Marker m;
      m.header.frame_id = frame_id_;
      m.header.stamp    = now;
      m.ns = "slot_rect";
      m.id = id_line++;
      m.type   = Marker::LINE_LIST;
      m.action = Marker::ADD;
      m.scale.x = line_width_;
      m.color.r = 0.1f; m.color.g = 1.0f; m.color.b = 0.1f; m.color.a = 1.0f;
      m.pose.orientation.w = 1.0;

      auto add_edge = [&](int i, int j){
        Point a, b;
        a.x = box[i].x; a.y = box[i].y; a.z = 0.0;
        b.x = box[j].x; b.y = box[j].y; b.z = 0.0;
        m.points.push_back(a); m.points.push_back(b);
      };
      add_edge(0,1); add_edge(1,2); add_edge(2,3); add_edge(3,0);
      out.markers.push_back(m);

      // 텍스트(폭 x 길이)
      auto len=[](cv::Point2f a, cv::Point2f b){ return std::hypot(b.x-a.x, b.y-a.y); };
      float e0 = len(box[0], box[1]);
      float e1 = len(box[1], box[2]);
      float w  = std::min(e0, e1);
      float h  = std::max(e0, e1);

      Point c; for (auto& q: box){ c.x += q.x; c.y += q.y; } c.x/=4.0; c.y/=4.0;
      Marker t;
      t.header.frame_id = frame_id_;
      t.header.stamp    = now;
      t.ns = "slot_text";
      t.id = id_text++;
      t.type   = Marker::TEXT_VIEW_FACING;
      t.action = Marker::ADD;
      t.scale.z = 0.35;
      t.color.r = 1.0f; t.color.g = 1.0f; t.color.b = 1.0f; t.color.a = 1.0f;
      t.pose.position = c;
      char buf[64]; std::snprintf(buf, sizeof(buf), "%.2fm x %.2fm", w, h);
      t.text = buf;
      out.markers.push_back(t);
    }

    pub_->publish(out);
  }

  // params
  std::string frame_id_, in_topic_;
  double w_nom_, h_nom_, w_tol_, h_tol_, line_width_;

  // io
  rclcpp::Subscription<MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr    pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RectFitterNode>());
  rclcpp::shutdown();
  return 0;
}
