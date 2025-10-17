#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <optional>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdio>

using std::placeholders::_1;

struct Line2D {
  Eigen::Vector2f p0;
  Eigen::Vector2f u;
  float s_min, s_max;
  float smin_raw, smax_raw;
  std::vector<float> s_points;
};

static float wrapAngle(float a){
  while (a >= M_PI) a -= 2.f*M_PI;
  while (a < -M_PI) a += 2.f*M_PI;
  return a;
}
static float rad2deg(float r){ return r*180.f/static_cast<float>(M_PI); }

class RectFitterNode : public rclcpp::Node {
public:
  RectFitterNode() : Node("rect_fitter_node") {
    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "/parking_group_markers", 10, std::bind(&RectFitterNode::onMarkers, this, _1));
    pub_lines_ = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/lines", 10);
    pub_rects_ = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/rects", 10);

    // params
    extension_len_ = declare_parameter("extension_len", 1.0);
    extension_len_ext5_ = declare_parameter("extension_len_ext5", 2.5); // ★ 5m 추가
    angle_bin_tol_deg_ = declare_parameter("angle_bin_tol_deg", 5.0);
    min_side_len_ = declare_parameter("min_side_len", 0.8);
    max_side_len_ = declare_parameter("max_side_len", 6.0);
    near_parallel_dot_thresh_ = std::cos((90.0 - angle_bin_tol_deg_) * M_PI/180.0);
    near_colinear_dot_thresh_ = std::cos((angle_bin_tol_deg_) * M_PI/180.0);
    support_tol_dist_ = declare_parameter("support_tol_dist", 0.15);
    min_support_pts_per_side_ = declare_parameter("min_support_pts_per_side", 1);
    edge_min_len_ = declare_parameter("edge_min_len", 1.2);
    max_edge_len_strict_ = declare_parameter("max_edge_len_strict", 5.0); // ★ 기본 5 m

    RCLCPP_INFO(get_logger(), "✅ rect_fitter_node started.");
  }

private:
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lines_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rects_;

  double extension_len_, extension_len_ext5_;
  double angle_bin_tol_deg_;
  double min_side_len_, max_side_len_;
  double near_parallel_dot_thresh_, near_colinear_dot_thresh_;
  double support_tol_dist_;
  int    min_support_pts_per_side_;
  double edge_min_len_;
  int last_rect_label_count_ = 0;
  int last_rect_label_count_ext5_ = 0; // ★ 빨강 라벨 잔상 제거용
  double max_edge_len_strict_; 

  struct Group { std::string ns; int id; std::vector<Eigen::Vector2f> pts; };

  static std::optional<Line2D>
  fitLinePCA(const std::vector<Eigen::Vector2f>& pts, double extension_len) {
    if (pts.size() < 2) return std::nullopt;
    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    for (auto& p : pts) mean += p;
    mean /= static_cast<float>(pts.size());
    Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
    for (auto& p : pts) {
      Eigen::Vector2f d = p - mean;
      C += d * d.transpose();
    }
    if (pts.size() > 1) C /= static_cast<float>(pts.size()-1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
    Eigen::Vector2f u = es.eigenvectors().col(1);
    if (u.norm() < 1e-9f) return std::nullopt;
    u.normalize();

    float smin = +1e9f, smax = -1e9f;
    std::vector<float> svals; svals.reserve(pts.size());
    for (auto& p : pts) {
      float s = (p - mean).dot(u);
      svals.push_back(s);
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }
    Line2D L;
    L.p0 = mean; L.u = u;
    L.smin_raw = smin; L.smax_raw = smax;
    L.s_min = smin - static_cast<float>(extension_len);
    L.s_max = smax + static_cast<float>(extension_len);
    L.s_points = std::move(svals);
    return L;
  }

  static bool isPureExtensionSegment(const Line2D& L, float sa, float sb, float eps = 1e-5f) {
    if (sa > sb) std::swap(sa, sb);
    float inter_len = std::min(sb, L.smax_raw) - std::max(sa, L.smin_raw);
    return inter_len <= eps; // 원본 구간과 교집합 길이 ≈ 0 → 순수 연장
  }

  static bool intersectLinesST(const Line2D& A, const Line2D& B,
                               Eigen::Vector2f& out, float& s_out, float& t_out,
                               bool clamp_to_segments = true) {
    const auto& p0 = A.p0; const auto& u = A.u;
    const auto& q0 = B.p0; const auto& v = B.u;
    float den = u.x()*v.y() - u.y()*v.x();
    if (std::fabs(den) < 1e-9f) return false;
    Eigen::Vector2f w = q0 - p0;
    float s = (w.x()*v.y() - w.y()*v.x()) / den;
    float t = (w.x()*u.y() - w.y()*u.x()) / den;
    if (clamp_to_segments) {
      if (s < A.s_min || s > A.s_max || t < B.s_min || t > B.s_max) return false;
    }
    out = p0 + s*u; s_out = s; t_out = t; return true;
  }

  int countSupportOnSegment(const Line2D& L, float sa, float sb,
                            double /*tol_dist*/, int min_pts_required) {
    if (sa > sb) std::swap(sa, sb);
    float ia = std::max(sa, L.smin_raw);
    float ib = std::min(sb, L.smax_raw);
    if (ia > ib) return 0;
    int cnt = 0;
    for (float s : L.s_points) {
      if (s < ia || s > ib) continue;
      if (++cnt >= min_pts_required) break;
    }
    return cnt;
  }

  // 사각형 구조
  struct Rect { Eigen::Vector2f c[4]; float w, h; };

  // 유틸: 평행/직교
  bool is_orthogonal(const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return std::fabs(a.normalized().dot(b.normalized())) <= near_parallel_dot_thresh_;
  }
  bool is_parallel(const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return std::fabs(a.normalized().dot(b.normalized())) >= near_colinear_dot_thresh_;
  }

  // 사각형 후보 생성 (필터 강도 토글)
  std::vector<Rect> generateRects(const std::vector<Line2D>& linesX,
                                  const std::vector<Line2D>& linesY,
                                  bool require_support,
                                  int support_k = 4,
                                  bool strict_rect = false,
                                  bool require_pure_extension_one_edge = false,
                                  bool require_support_for_non_pure_edges = false) {
    std::vector<Rect> rects;
    for (size_t i=0;i<linesX.size();++i){
      for (size_t j=i+1;j<linesX.size();++j){
        if (!is_parallel(linesX[i].u, linesX[j].u)) continue;
        for (size_t k=0;k<linesY.size();++k){
          for (size_t l=k+1;l<linesY.size();++l){
            if (!is_parallel(linesY[k].u, linesY[l].u)) continue;

            Eigen::Vector2f P00,P01,P10,P11;
            float s_i_k,s_i_l,s_j_k,s_j_l, t_k_i,t_l_i,t_k_j,t_l_j;
            if (!intersectLinesST(linesX[i], linesY[k], P00, s_i_k, t_k_i)) continue;
            if (!intersectLinesST(linesX[i], linesY[l], P01, s_i_l, t_l_i)) continue;
            if (!intersectLinesST(linesX[j], linesY[k], P10, s_j_k, t_k_j)) continue;
            if (!intersectLinesST(linesX[j], linesY[l], P11, s_j_l, t_l_j)) continue;

            int sup_e0 = countSupportOnSegment(linesX[i], s_i_k, s_i_l, support_tol_dist_, min_support_pts_per_side_);
            int sup_e2 = countSupportOnSegment(linesX[j], s_j_k, s_j_l, support_tol_dist_, min_support_pts_per_side_);
            int sup_e1 = countSupportOnSegment(linesY[k], t_k_i, t_k_j, support_tol_dist_, min_support_pts_per_side_);
            int sup_e3 = countSupportOnSegment(linesY[l], t_l_i, t_l_j, support_tol_dist_, min_support_pts_per_side_);

            // 각 변이 '순수 연장'인지 판정
            bool pure0 = isPureExtensionSegment(linesX[i], s_i_k, s_i_l);
            bool pure2 = isPureExtensionSegment(linesX[j], s_j_k, s_j_l);
            bool pure1 = isPureExtensionSegment(linesY[k], t_k_i, t_k_j);
            bool pure3 = isPureExtensionSegment(linesY[l], t_l_i, t_l_j);

            // 규칙 A: 최소 한 변은 순수 연장부
            if (require_pure_extension_one_edge) {
              if (!(pure0 || pure1 || pure2 || pure3)) continue;
            }

            // 규칙 B: 순수 연장이 아닌 변들은 '각각' 지지점이 기준 이상이어야 함
            if (require_support_for_non_pure_edges) {
              if (!pure0 && sup_e0 < min_support_pts_per_side_) continue;
              if (!pure1 && sup_e1 < min_support_pts_per_side_) continue;
              if (!pure2 && sup_e2 < min_support_pts_per_side_) continue;
              if (!pure3 && sup_e3 < min_support_pts_per_side_) continue;
            }

            if (require_support) {
              if (sup_e0 < min_support_pts_per_side_ ||
                  sup_e1 < min_support_pts_per_side_ ||
                  sup_e2 < min_support_pts_per_side_ ||
                  sup_e3 < min_support_pts_per_side_) continue;
            } else {
              // k-of-4 지지 허용 (연장 위주)
              int supported = (sup_e0>=min_support_pts_per_side_) + (sup_e1>=min_support_pts_per_side_) +
                              (sup_e2>=min_support_pts_per_side_) + (sup_e3>=min_support_pts_per_side_);
              if (supported < support_k) continue; // 예: 0~1도 허용하고 싶다면 더 낮춰도 됨
            }

            Eigen::Vector2f center = 0.25f*(P00+P01+P10+P11);
            std::array<Eigen::Vector2f,4> vs = {P00,P01,P11,P10};
            std::sort(vs.begin(), vs.end(), [&](const auto& A, const auto& B){
              float a = std::atan2(A.y()-center.y(), A.x()-center.x());
              float b = std::atan2(B.y()-center.y(), B.x()-center.x());
              return wrapAngle(a) < wrapAngle(b);
            });

            float e0 = (vs[1]-vs[0]).norm();
            float e1 = (vs[2]-vs[1]).norm();
            float e2 = (vs[3]-vs[2]).norm();
            float e3 = (vs[0]-vs[3]).norm();

            // ★ 한 변이라도 5 m(파라미터) 초과면 탈락
            if (e0 > max_edge_len_strict_ || e1 > max_edge_len_strict_ ||
                e2 > max_edge_len_strict_ || e3 > max_edge_len_strict_) {
              continue;
            }

            if (e0 < edge_min_len_ || e1 < edge_min_len_ || e2 < edge_min_len_ || e3 < edge_min_len_) continue;

            auto approx_eq = [](float a, float b){ return std::fabs(a-b) <= 0.3f*std::max(1.0f,std::max(a,b)); };
            bool opp_ok = approx_eq(e0,e2) && approx_eq(e1,e3);
            float w = 0.5f*(e0+e2), h = 0.5f*(e1+e3);
            if (!opp_ok) continue;
            if (w < min_side_len_ || h < min_side_len_) continue;
            if (w > max_side_len_ || h > max_side_len_) continue;

            Eigen::Vector2f v01=(vs[1]-vs[0]).normalized();
            Eigen::Vector2f v12=(vs[2]-vs[1]).normalized();
            if (!is_orthogonal(v01, v12)) continue;

            Rect R; R.c[0]=vs[0]; R.c[1]=vs[1]; R.c[2]=vs[2]; R.c[3]=vs[3]; R.w=w; R.h=h;
            rects.push_back(R);
          }
        }
      }
    }
    return rects;
  }

  // 초록과 빨강이 겹치는지 판정: 중심 거리 + 크기 유사
  static bool rectsOverlapLoose(const Rect& A, const Rect& B){
    auto center = [](const Rect& R){
      return 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
    };
    Eigen::Vector2f ca = center(A), cb = center(B);
    float dc = (ca-cb).norm();
    // 크기 유사 (±25%), 중심 0.5m 이내면 같은 것으로 간주
    bool size_sim = (std::fabs(A.w-B.w) <= 0.25f*std::max(A.w,B.w)) &&
                    (std::fabs(A.h-B.h) <= 0.25f*std::max(A.h,B.h));
    return (dc < 0.5f) && size_sim;
  }

  void onMarkers(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
    std::vector<Group> groupsX, groupsY;
    std_msgs::msg::Header header;
    if (!msg->markers.empty()) header = msg->markers.front().header;

    for (const auto& m : msg->markers) {
      if (m.type != visualization_msgs::msg::Marker::POINTS) continue;
      if (m.action == visualization_msgs::msg::Marker::DELETE) continue;
      if (m.points.size() < 2) continue;
      Group g; g.ns=m.ns; g.id=m.id; g.pts.reserve(m.points.size());
      for (const auto& P : m.points) g.pts.emplace_back(P.x,P.y);
      if (m.ns=="groupX_pts") groupsX.push_back(std::move(g));
      else if (m.ns=="groupY_pts") groupsY.push_back(std::move(g));
    }

    // 1) 기본 연장 길이로 선 핏팅 (초록)
    std::vector<Line2D> linesX, linesY;
    for (auto& g: groupsX){ if (auto L=fitLinePCA(g.pts, extension_len_)) linesX.push_back(*L); }
    for (auto& g: groupsY){ if (auto L=fitLinePCA(g.pts, extension_len_)) linesY.push_back(*L); }

    // 2) 5m 연장으로 선 핏팅 (빨강 후보용)
    std::vector<Line2D> linesX5, linesY5;
    for (auto& g: groupsX){ if (auto L=fitLinePCA(g.pts, extension_len_ext5_)) linesX5.push_back(*L); }
    for (auto& g: groupsY){ if (auto L=fitLinePCA(g.pts, extension_len_ext5_)) linesY5.push_back(*L); }

    // 선 시각화(초록=기존, 하늘=기존 Y)
    visualization_msgs::msg::MarkerArray line_arr;
    {
      visualization_msgs::msg::Marker mx;
      mx.header=header; mx.ns="rect_lines_x"; mx.id=0;
      mx.type=visualization_msgs::msg::Marker::LINE_LIST;
      mx.action=visualization_msgs::msg::Marker::ADD;
      mx.scale.x=0.06; mx.color.r=1.0f; mx.color.g=0.6f; mx.color.b=0.1f; mx.color.a=0.4f;
      for (auto& L: linesX){
        Eigen::Vector2f A=L.p0+L.s_min*L.u, B=L.p0+L.s_max*L.u;
        geometry_msgs::msg::Point pA,pB; pA.x=A.x(); pA.y=A.y(); pA.z=0; pB.x=B.x(); pB.y=B.y(); pB.z=0;
        mx.points.push_back(pA); mx.points.push_back(pB);
      }
      line_arr.markers.push_back(mx);

      visualization_msgs::msg::Marker my;
      my.header=header; my.ns="rect_lines_y"; my.id=0;
      my.type=visualization_msgs::msg::Marker::LINE_LIST;
      my.action=visualization_msgs::msg::Marker::ADD;
      my.scale.x=0.06; my.color.r=0.1f; my.color.g=0.7f; my.color.b=1.0f; my.color.a=0.4f;
      for (auto& L: linesY){
        Eigen::Vector2f A=L.p0+L.s_min*L.u, B=L.p0+L.s_max*L.u;
        geometry_msgs::msg::Point pA,pB; pA.x=A.x(); pA.y=A.y(); pA.z=0; pB.x=B.x(); pB.y=B.y(); pB.z=0;
        my.points.push_back(pA); my.points.push_back(pB);
      }
      line_arr.markers.push_back(my);
    }
    pub_lines_->publish(line_arr);

    // ---------- 사각형 찾기 ----------
    // A) 초록(기본 연장, 지지 필수)
    auto green_rects = generateRects(linesX, linesY, /*require_support=*/true);

    // B) 빨강(5m 연장, 지지 완화: k-of-4, 여기선 k=0로 완전 허용도 가능)
    auto red_candidates = generateRects(linesX5, linesY5,
                                        /*require_support=*/false,    // 내부에서 k-of-4 안 씀
                                        /*support_k=*/0,              // 의미없음
                                        /*strict_rect=*/true,         // (권장) 직교/평행 더 타이트
                                        /*require_pure_extension_one_edge=*/true,       // 규칙 A
                                        /*require_support_for_non_pure_edges=*/true);    // 규칙 B
    // C) 겹치지 않는 빨강만 남기기
    std::vector<Rect> red_rects;
    for (const auto& rr : red_candidates){
      bool overlap=false;
      for (const auto& gg : green_rects){
        if (rectsOverlapLoose(gg, rr)) { overlap=true; break; }
      }
      if (!overlap) red_rects.push_back(rr);
    }

    // ---------- 시각화 ----------
    visualization_msgs::msg::MarkerArray rect_arr;

    // 초록 LINE_LIST (edges)
    {
      visualization_msgs::msg::Marker m;
      m.header=header; m.ns="rects"; m.id=0;
      m.type=visualization_msgs::msg::Marker::LINE_LIST;
      m.scale.x=0.1;
      m.color.r=0.2f; m.color.g=1.0f; m.color.b=0.2f; m.color.a=0.98f;
      for (auto& R: green_rects){
        for (int i=0;i<4;i++){
          const auto& A=R.c[i]; const auto& B=R.c[(i+1)%4];
          geometry_msgs::msg::Point pA,pB; pA.x=A.x(); pA.y=A.y(); pA.z=0; pB.x=B.x(); pB.y=B.y(); pB.z=0;
          m.points.push_back(pA); m.points.push_back(pB);
        }
      }
      m.action = m.points.empty()? visualization_msgs::msg::Marker::DELETE
                                 : visualization_msgs::msg::Marker::ADD;
      rect_arr.markers.push_back(m);
    }
    // 초록 라벨
    {
      int rid=0;
      for (auto& R: green_rects){
        visualization_msgs::msg::Marker t;
        t.header=header; t.ns="rect_labels"; t.id=rid;
        t.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action=visualization_msgs::msg::Marker::ADD;
        Eigen::Vector2f center = 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
        t.pose.position.x=center.x(); t.pose.position.y=center.y(); t.pose.position.z=0.1;
        t.scale.z=0.35;
        t.color.r=1; t.color.g=1; t.color.b=1; t.color.a=1;
        char buf[128]; std::snprintf(buf,sizeof(buf),"w=%.2f, h=%.2f",R.w,R.h);
        t.text=buf;
        rect_arr.markers.push_back(t);
        ++rid;
      }
      for (int i=(int)green_rects.size(); i<last_rect_label_count_; ++i){
        visualization_msgs::msg::Marker del;
        del.header=header; del.ns="rect_labels"; del.id=i;
        del.action=visualization_msgs::msg::Marker::DELETE;
        rect_arr.markers.push_back(del);
      }
      last_rect_label_count_ = (int)green_rects.size();
      RCLCPP_INFO(get_logger(), "주차 블가능 공간 개수: %zu", green_rects.size());
    }

    // 빨강 LINE_LIST (edges, 겹치지 않는 것만)
    {
      visualization_msgs::msg::Marker m;
      m.header=header; m.ns="rects_ext5"; m.id=0;
      m.type=visualization_msgs::msg::Marker::LINE_LIST;
      m.scale.x=0.1;
      m.color.r=1.0f; m.color.g=0.2f; m.color.b=0.2f; m.color.a=0.98f; // 빨강
      for (auto& R: red_rects){
        for (int i=0;i<4;i++){
          const auto& A=R.c[i]; const auto& B=R.c[(i+1)%4];
          geometry_msgs::msg::Point pA,pB; pA.x=A.x(); pA.y=A.y(); pA.z=0; pB.x=B.x(); pB.y=B.y(); pB.z=0;
          m.points.push_back(pA); m.points.push_back(pB);
        }
      }
      m.action = m.points.empty()? visualization_msgs::msg::Marker::DELETE
                                 : visualization_msgs::msg::Marker::ADD;
      rect_arr.markers.push_back(m);
    }
    // 빨강 라벨
    {
      int rid=0;
      for (auto& R: red_rects){
        visualization_msgs::msg::Marker t;
        t.header=header; t.ns="rect_labels_ext5"; t.id=rid;
        t.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action=visualization_msgs::msg::Marker::ADD;
        Eigen::Vector2f center = 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
        t.pose.position.x=center.x(); t.pose.position.y=center.y(); t.pose.position.z=0.1;
        t.scale.z=0.35;
        t.color.r=1; t.color.g=0.8; t.color.b=0.8; t.color.a=1;
        char buf[128]; std::snprintf(buf,sizeof(buf),"[ext5] w=%.2f, h=%.2f",R.w,R.h);
        t.text=buf;
        rect_arr.markers.push_back(t);
        ++rid;
      }
      for (int i=(int)red_rects.size(); i<last_rect_label_count_ext5_; ++i){
        visualization_msgs::msg::Marker del;
        del.header=header; del.ns="rect_labels_ext5"; del.id=i;
        del.action=visualization_msgs::msg::Marker::DELETE;
        rect_arr.markers.push_back(del);
      }
      last_rect_label_count_ext5_ = (int)red_rects.size();
      RCLCPP_INFO(get_logger(), "주차 가능 공간 개수: %zu", red_rects.size());
    }

    pub_rects_->publish(rect_arr);
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RectFitterNode>());
  rclcpp::shutdown();
  return 0;
}
