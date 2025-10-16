#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <optional>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;

struct Line2D {
  // 2D 직선: 기준점 p0, 단위방향 u (|u|=1)
  Eigen::Vector2f p0;
  Eigen::Vector2f u;
  float s_min;     // 확장 후
  float s_max;     // 확장 후
  float smin_raw;  // 확장 전 (원본)
  float smax_raw;  // 확장 전 (원본)
  std::vector<float> s_points; // 이 선을 만든 원본 포인트들의 투영값 목록
};

static float wrapAngle(float a) {
  // [-pi, pi)
  while (a >= M_PI) a -= 2.f * M_PI;
  while (a < -M_PI) a += 2.f * M_PI;
  return a;
}
static float rad2deg(float r){ return r * 180.f / static_cast<float>(M_PI); }

class RectFitterNode : public rclcpp::Node {
public:
  RectFitterNode() : Node("rect_fitter_node") {
    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "/parking_group_markers", 10, std::bind(&RectFitterNode::onMarkers, this, _1));

    pub_lines_ = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/lines", 10);
    pub_rects_ = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/rects", 10);

    // 파라미터(원하면 ros2 param으로 바꿔도 됨)
    extension_len_ = declare_parameter("extension_len", 3.0);             // 각 끝단 3 m 확장
    angle_bin_tol_deg_ = declare_parameter("angle_bin_tol_deg", 21.0);    // 직교/평행 판단 여유
    min_side_len_ = declare_parameter("min_side_len", 0.8);               // 사각형 최소변
    max_side_len_ = declare_parameter("max_side_len", 10.0);              // 사각형 최대변
    near_parallel_dot_thresh_ = std::cos((90.0 - angle_bin_tol_deg_) * M_PI/180.0); // 직교 판단용
    near_colinear_dot_thresh_ = std::cos((angle_bin_tol_deg_) * M_PI/180.0);        // 평행 판단용
    support_tol_dist_ = declare_parameter("support_tol_dist", 0.15); // 15cm
    min_support_pts_per_side_ = declare_parameter("min_support_pts_per_side", 2);
    edge_min_len_ = declare_parameter("edge_min_len", 2.0); // 각 변 최소 2m
    RCLCPP_INFO(get_logger(), "✅ rect_fitter_node started.");
  }

private:
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lines_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rects_;

  double extension_len_;
  double angle_bin_tol_deg_;
  double min_side_len_, max_side_len_;
  double near_parallel_dot_thresh_, near_colinear_dot_thresh_;
  double support_tol_dist_;              // 선으로부터 허용 근접거리 (m)
  int    min_support_pts_per_side_;      // 변 하나당 최소 지지 포인트 수
  double edge_min_len_; 

  struct Group {
    std::string ns;
    int id;
    std::vector<Eigen::Vector2f> pts;  // z는 무시하고 2D로 처리
  };

  static std::optional<Line2D> 
  fitLinePCA(const std::vector<Eigen::Vector2f>& pts, double extension_len) {
    if (pts.size() < 2) return std::nullopt;

    // 평균
    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    for (auto& p : pts) mean += p;
    mean /= static_cast<float>(pts.size());

    // 공분산
    Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
    for (auto& p : pts) {
      Eigen::Vector2f d = p - mean;
      C += d * d.transpose();
    }
    if (pts.size() > 1) C /= static_cast<float>(pts.size()-1);

    // 주성분(최대 고유값의 고유벡터)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
    Eigen::Vector2f u = es.eigenvectors().col(1); // col(1): 큰 고유값 쪽
    if (u.norm() < 1e-9f) return std::nullopt;
    u.normalize();

    // 투영 범위
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

  // ---- 교차에서 s,t도 리턴하도록 확장 ----
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
    out = p0 + s*u;
    s_out = s; t_out = t;
    return true;
  }

  // ---- 특정 '변' 구간의 지지 포인트 개수 세기 ----
  //  * 변은 한 '선'(Line2D) 위의 s-구간 [sa, sb] 로 정의
  //  * 원본 구간 [smin_raw, smax_raw] 안에 실제 포인트가 2개 이상 존재해야 함
  int countSupportOnSegment(const Line2D& L, float sa, float sb,
                            double tol_dist, int min_pts_required) {
    if (sa > sb) std::swap(sa, sb);

    // 변 구간과 원본 구간의 교집합만 카운트 (연장부 제외)
    float ia = std::max(sa, L.smin_raw);
    float ib = std::min(sb, L.smax_raw);
    if (ia > ib) return 0;

    // s-구간 안에 들어오는 원본 포인트 수 세기
    int cnt = 0;
    for (float s : L.s_points) {
      if (s < ia || s > ib) continue;
      // 수직 거리 판정 (P = p0 + s*u → 원 포인트까지의 직교거리 필요하면 여기서 계산 가능)
      // L.s_points는 투영만 들고 있으니, 보수적으로 tol은 s-조건으로만 판단.
      // 더 엄격히 하려면 pts도 저장해서 실제 거리 체크 가능.
      cnt++;
      if (cnt >= min_pts_required) break;
    }
    return cnt;
  }

  static float dotAbs(const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return std::fabs(a.normalized().dot(b.normalized()));
  }

  void onMarkers(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
    // 입력: /parking_group_markers 에서 넘어온 POINTS들
    // ns "groupX_pts" / "groupY_pts" 각각에 여러 id가 있음.
    std::vector<Group> groupsX, groupsY;

    // 프레임/타임스탬프는 첫 마커 기준
    std_msgs::msg::Header header;
    if (!msg->markers.empty()) header = msg->markers.front().header;

    for (const auto& m : msg->markers) {
      if (m.type != visualization_msgs::msg::Marker::POINTS) continue;
      if (m.action == visualization_msgs::msg::Marker::DELETE) continue;
      if (m.points.size() < 2) continue;

      Group g; g.ns = m.ns; g.id = m.id;
      g.pts.reserve(m.points.size());
      for (const auto& P : m.points) g.pts.emplace_back(P.x, P.y);

      if (m.ns == "groupX_pts") groupsX.push_back(std::move(g));
      else if (m.ns == "groupY_pts") groupsY.push_back(std::move(g));
      else {
        // 다른 ns는 무시
      }
    }

    // 각 그룹을 선으로 핏팅 + 3m 확장
    std::vector<Line2D> linesX, linesY;
    linesX.reserve(groupsX.size());
    linesY.reserve(groupsY.size());

    for (auto& g : groupsX) {
      auto L = fitLinePCA(g.pts, extension_len_);
      if (L) linesX.push_back(*L);
    }
    for (auto& g : groupsY) {
      auto L = fitLinePCA(g.pts, extension_len_);
      if (L) linesY.push_back(*L);
    }

    // 시각화: 확장 선분들
    visualization_msgs::msg::MarkerArray line_arr;
    {
      visualization_msgs::msg::Marker mx;
      mx.header = header; mx.ns = "rect_lines_x"; mx.id = 0;
      mx.type = visualization_msgs::msg::Marker::LINE_LIST;
      mx.action = visualization_msgs::msg::Marker::ADD;
      mx.scale.x = 0.06;
      mx.color.r = 1.0f; mx.color.g = 0.6f; mx.color.b = 0.1f; mx.color.a = 0.6f;
      for (auto& L : linesX) {
        Eigen::Vector2f A = L.p0 + L.s_min * L.u;
        Eigen::Vector2f B = L.p0 + L.s_max * L.u;
        geometry_msgs::msg::Point pA, pB; pA.x=A.x(); pA.y=A.y(); pA.z=0.0; pB.x=B.x(); pB.y=B.y(); pB.z=0.0;
        mx.points.push_back(pA); mx.points.push_back(pB);
      }
      line_arr.markers.push_back(mx);

      visualization_msgs::msg::Marker my;
      my.header = header; my.ns = "rect_lines_y"; my.id = 0;
      my.type = visualization_msgs::msg::Marker::LINE_LIST;
      my.action = visualization_msgs::msg::Marker::ADD;
      my.scale.x = 0.06;
      my.color.r = 0.1f; my.color.g = 0.7f; my.color.b = 1.0f; my.color.a = 0.6f;
      for (auto& L : linesY) {
        Eigen::Vector2f A = L.p0 + L.s_min * L.u;
        Eigen::Vector2f B = L.p0 + L.s_max * L.u;
        geometry_msgs::msg::Point pA, pB; pA.x=A.x(); pA.y=A.y(); pA.z=0.0; pB.x=B.x(); pB.y=B.y(); pB.z=0.0;
        my.points.push_back(pA); my.points.push_back(pB);
      }
      line_arr.markers.push_back(my);
    }
    pub_lines_->publish(line_arr);

    // 직교성/평행성 체크를 위해 방향 벡터 정리
    auto is_orthogonal = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
      // |dot(a,b)| <= cos(90 - tol)
      return std::fabs(a.normalized().dot(b.normalized())) <= near_parallel_dot_thresh_;
    };
    auto is_parallel = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
      return std::fabs(a.normalized().dot(b.normalized())) >= near_colinear_dot_thresh_;
    };

    // 사각형 후보 생성:
    //   X계에서 "거의 평행한" 두 선을 고르고,
    //   Y계에서 "거의 평행한" 두 선을 고른 후,
    //   네 교차점을 만들어 사각형 구성.
    struct Rect {
      Eigen::Vector2f c[4]; // 시계방향 정렬된 꼭짓점
      float w, h;           // 인접 변 길이
    };
    std::vector<Rect> rects;

    for (size_t i=0;i<linesX.size();++i){
      for (size_t j=i+1;j<linesX.size();++j){
        if (!is_parallel(linesX[i].u, linesX[j].u)) continue; // X끼리 평행
        for (size_t k=0;k<linesY.size();++k){
          for (size_t l=k+1;l<linesY.size();++l){
            if (!is_parallel(linesY[k].u, linesY[l].u)) continue; // Y끼리 평행
            // 교차점 4개: (Xi,Yk), (Xi,Yl), (Xj,Yk), (Xj,Yl)
            Eigen::Vector2f P00, P01, P10, P11;
            float s_i_k, s_i_l, s_j_k, s_j_l; // X-라인에서의 s
            float t_k_i, t_l_i, t_k_j, t_l_j; // Y-라인에서의 t

            if (!intersectLinesST(linesX[i], linesY[k], P00, s_i_k, t_k_i)) continue;
            if (!intersectLinesST(linesX[i], linesY[l], P01, s_i_l, t_l_i)) continue;
            if (!intersectLinesST(linesX[j], linesY[k], P10, s_j_k, t_k_j)) continue;
            if (!intersectLinesST(linesX[j], linesY[l], P11, s_j_l, t_l_j)) continue;

            // ✅ 각 변의 지지 포인트 개수 확인
            int sup_e0 = countSupportOnSegment(linesX[i], s_i_k, s_i_l, support_tol_dist_, min_support_pts_per_side_); // X_i 위 변
            int sup_e2 = countSupportOnSegment(linesX[j], s_j_k, s_j_l, support_tol_dist_, min_support_pts_per_side_); // X_j 위 변
            int sup_e1 = countSupportOnSegment(linesY[k], t_k_i, t_k_j, support_tol_dist_, min_support_pts_per_side_); // Y_k 위 변
            int sup_e3 = countSupportOnSegment(linesY[l], t_l_i, t_l_j, support_tol_dist_, min_support_pts_per_side_); // Y_l 위 변

            // 하나라도 2 미만이면 → 연장만으로 만든 변 or 연장+한 점뿐인 변 → 사각형 탈락
            if (sup_e0 < min_support_pts_per_side_ ||
                sup_e1 < min_support_pts_per_side_ ||
                sup_e2 < min_support_pts_per_side_ ||
                sup_e3 < min_support_pts_per_side_) {
              continue;
            }

            // 사각형 정렬 (대충 중심 기준 각도 정렬)
            Eigen::Vector2f center = 0.25f*(P00+P01+P10+P11);
            std::array<Eigen::Vector2f,4> vs = {P00,P01,P11,P10}; // 대략적 인접 배치
            std::sort(vs.begin(), vs.end(), [&](const auto& A, const auto& B){
              float a = std::atan2(A.y()-center.y(), A.x()-center.x());
              float b = std::atan2(B.y()-center.y(), B.x()-center.x());
              return wrapAngle(a) < wrapAngle(b);
            });

            // 변 길이
            float e0 = (vs[1]-vs[0]).norm();
            float e1 = (vs[2]-vs[1]).norm();
            float e2 = (vs[3]-vs[2]).norm();
            float e3 = (vs[0]-vs[3]).norm();

            // ✅ 각 변이 모두 2m 이상이어야 통과
            if (e0 < edge_min_len_ || e1 < edge_min_len_ || e2 < edge_min_len_ || e3 < edge_min_len_) {
              continue;
            }

            // 대충 직사각형인지(대변 길이 유사) & 변 길이 범위
            auto approx_eq = [](float a, float b){ return std::fabs(a-b) <= 0.3f * std::max(1.0f, std::max(a,b)); };
            bool opp_ok = approx_eq(e0, e2) && approx_eq(e1, e3);
            float w = 0.5f * (e0 + e2);
            float h = 0.5f * (e1 + e3);

            if (!opp_ok) continue;
            if (w < min_side_len_ || h < min_side_len_) continue;
            if (w > max_side_len_ || h > max_side_len_) continue;

            // 직교성 검사(이웃 변) : u·v ≈ 0
            Eigen::Vector2f v01 = (vs[1]-vs[0]).normalized();
            Eigen::Vector2f v12 = (vs[2]-vs[1]).normalized();
            if (!is_orthogonal(v01, v12)) continue;

            Rect R; R.c[0]=vs[0]; R.c[1]=vs[1]; R.c[2]=vs[2]; R.c[3]=vs[3]; R.w=w; R.h=h;
            rects.push_back(R);
          }
        }
      }
    }

    // 시각화: 사각형들
    visualization_msgs::msg::MarkerArray rect_arr;
    {
      // 모든 사각형을 한 마커로 그려도 되고, 개별 id로 나눠도 됨.
      // 여기서는 보기 좋게 각 사각형을 개별 LINE_STRIP으로.
      int rid = 0;
      for (auto& R : rects) {
        visualization_msgs::msg::Marker m;
        m.header = header; m.ns = "rects"; m.id = rid++;
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.08;
        m.color.r = 0.2f; m.color.g = 1.0f; m.color.b = 0.2f; m.color.a = 0.98f;

        for (int i=0;i<4;++i){
          geometry_msgs::msg::Point P; P.x=R.c[i].x(); P.y=R.c[i].y(); P.z=0.0;
          m.points.push_back(P);
        }
        // 닫기
        geometry_msgs::msg::Point P0; P0.x=R.c[0].x(); P0.y=R.c[0].y(); P0.z=0.0;
        m.points.push_back(P0);

        rect_arr.markers.push_back(m);

        // 치수 표기(텍스트)
        visualization_msgs::msg::Marker t;
        t.header = header; t.ns = "rect_labels"; t.id = rid + 10000;
        t.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action = visualization_msgs::msg::Marker::ADD;
        Eigen::Vector2f center = 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
        t.pose.position.x = center.x(); t.pose.position.y = center.y(); t.pose.position.z = 0.1;
        t.scale.z = 0.4;
        t.color.r=1.0; t.color.g=1.0; t.color.b=1.0; t.color.a=1.0;
        char buf[128]; std::snprintf(buf, sizeof(buf), "w=%.2fm, h=%.2fm", R.w, R.h);
        t.text = buf;
        rect_arr.markers.push_back(t);
      }

      // 사각형이 하나도 없을 때 기존 잔여 id 삭제는, 간단히 이전 id 범위를 작게 유지하거나,
      // 매 프레임 id를 0..N-1로만 쓰는 현재 방식이면 자동으로 덮어써짐.
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
