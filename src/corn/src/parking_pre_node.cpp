#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <optional>
#include <algorithm>
#include <cmath>
#include <array>
#include <unordered_map>

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
static float rad2deg(float r){ return r * 180.f / static_cast<float>(M_PI); }

class ParkingPreNode : public rclcpp::Node {
public:
  ParkingPreNode() : Node("rect_fitter_goals_node") {
    sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "/parking_group_markers", 10, std::bind(&ParkingPreNode::onMarkers, this, _1));

    pub_lines_   = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/lines", 10);
    pub_rects_   = create_publisher<visualization_msgs::msg::MarkerArray>("/rect_fitter/rects", 10);

    // 🔁 변경: PoseArray → PoseStamped 로 목표를 보낼 퍼블리셔
    pub_goal_    = create_publisher<geometry_msgs::msg::PoseStamped>("/parking_goal", 10);

    // 🔁 변경: 시각화도 “삽입된 점만”
    pub_ins_vis_ = create_publisher<visualization_msgs::msg::MarkerArray>("/inserted_goal_markers", 10);

    // --- 파라미터 ---
    extension_len_         = declare_parameter("extension_len", 0.1);
    angle_bin_tol_deg_     = declare_parameter("angle_bin_tol_deg", 5.0);
    min_side_len_          = declare_parameter("min_side_len", 0.8);
    max_side_len_          = declare_parameter("max_side_len", 6.0);
    support_tol_dist_      = declare_parameter("support_tol_dist", 0.15);
    min_support_pts_side_  = declare_parameter("min_support_pts_per_side", 1);
    edge_min_len_          = declare_parameter("edge_min_len", 1.2);

    center_dedupe_radius_  = declare_parameter("center_dedupe_radius", 0.20);
    min_pts_for_line_      = declare_parameter("min_pts_for_line", 3);
    base_gap_floor_        = declare_parameter("base_gap_floor", 0.5);
    base_gap_ceil_         = declare_parameter("base_gap_ceil", 6.0);
    equalize_use_min_gap_  = declare_parameter("equalize_use_min_gap", true);
    spacing_insert_tol_    = declare_parameter("spacing_insert_tol", 0.25);

    near_parallel_dot_thresh_ = std::cos((90.0 - angle_bin_tol_deg_) * M_PI/180.0);
    near_colinear_dot_thresh_ = std::cos((angle_bin_tol_deg_) * M_PI/180.0);

    RCLCPP_INFO(get_logger(), "✅ Parking Pre Node Started.");
  }

private:
  // ---------------- ROS I/O ----------------
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lines_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rects_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_;                 // 🔁
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_ins_vis_;        // 🔁

  // ---------------- Params ----------------
  double extension_len_;
  double angle_bin_tol_deg_;
  double min_side_len_, max_side_len_;
  double near_parallel_dot_thresh_, near_colinear_dot_thresh_;
  double support_tol_dist_;
  int    min_support_pts_side_;
  double edge_min_len_;

  double center_dedupe_radius_;
  int    min_pts_for_line_;
  double base_gap_floor_, base_gap_ceil_;
  bool   equalize_use_min_gap_;
  double spacing_insert_tol_;

  // ---------------- State ----------------
  std::vector<Eigen::Vector2f> centers_accum_;
  std_msgs::msg::Header last_header_;
  int last_rect_label_count_ = 0;

  struct Group { std::string ns; int id; std::vector<Eigen::Vector2f> pts; };

  // -------- 공통 유틸 --------
  geometry_msgs::msg::Quaternion yawToQuat(double yaw){
    geometry_msgs::msg::Quaternion q;
    q.x=0.0; q.y=0.0; q.z=std::sin(yaw*0.5); q.w=std::cos(yaw*0.5);
    return q;
  }

  static std::optional<Line2D>
  fitLinePCA_pts(const std::vector<Eigen::Vector2f>& pts, double extension_len){
    if (pts.size() < 2) return std::nullopt;
    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    for (auto& p: pts) mean += p;
    mean /= (float)pts.size();

    Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
    for (auto& p: pts) {
      Eigen::Vector2f d = p - mean;
      C += d * d.transpose();
    }
    if (pts.size() > 1) C /= (float)(pts.size()-1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
    Eigen::Vector2f u = es.eigenvectors().col(1);
    if (u.norm() < 1e-9f) return std::nullopt;
    u.normalize();

    float smin=+1e9f, smax=-1e9f;
    std::vector<float> svals; svals.reserve(pts.size());
    for (auto& p: pts){
      float s = (p - mean).dot(u);
      svals.push_back(s);
      smin = std::min(smin, s);
      smax = std::max(smax, s);
    }
    Line2D L;
    L.p0 = mean; L.u = u;
    L.smin_raw = smin; L.smax_raw = smax;
    L.s_min = smin - (float)extension_len;
    L.s_max = smax + (float)extension_len;
    L.s_points = std::move(svals);
    return L;
  }

  bool is_parallel(const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return std::fabs(a.normalized().dot(b.normalized())) >= near_colinear_dot_thresh_;
  }
  bool is_orthogonal(const Eigen::Vector2f& a, const Eigen::Vector2f& b){
    return std::fabs(a.normalized().dot(b.normalized())) <= near_parallel_dot_thresh_;
  }

  struct Rect { Eigen::Vector2f c[4]; float w, h; };

  std::vector<Rect> generateGreenRects(
      const std::vector<Line2D>& linesX,
      const std::vector<Line2D>& linesY)
  {
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

            int sup_e0 = countSupportOnSegment(linesX[i], s_i_k, s_i_l, support_tol_dist_, min_support_pts_side_);
            int sup_e2 = countSupportOnSegment(linesX[j], s_j_k, s_j_l, support_tol_dist_, min_support_pts_side_);
            int sup_e1 = countSupportOnSegment(linesY[k], t_k_i, t_k_j, support_tol_dist_, min_support_pts_side_);
            int sup_e3 = countSupportOnSegment(linesY[l], t_l_i, t_l_j, support_tol_dist_, min_support_pts_side_);
            if (sup_e0 < min_support_pts_side_ || sup_e1 < min_support_pts_side_ ||
                sup_e2 < min_support_pts_side_ || sup_e3 < min_support_pts_side_) continue;

            Eigen::Vector2f center = 0.25f*(P00+P01+P10+P11);
            std::array<Eigen::Vector2f,4> vs = {P00,P01,P11,P10};
            std::sort(vs.begin(), vs.end(), [&](const auto& A, const auto& B){
              float a = std::atan2(A.y()-center.y(), A.x()-center.x());
              float b = std::atan2(B.y()-center.y(), B.x()-center.x());
              return wrapAngle(a) < wrapAngle(b);
            });

            float e0=(vs[1]-vs[0]).norm(), e1=(vs[2]-vs[1]).norm();
            float e2=(vs[3]-vs[2]).norm(), e3=(vs[0]-vs[3]).norm();
            if (e0 < edge_min_len_ || e1 < edge_min_len_ || e2 < edge_min_len_ || e3 < edge_min_len_) continue;

            auto approx_eq = [](float a,float b){ return std::fabs(a-b) <= 0.3f*std::max(1.0f,std::max(a,b)); };
            bool opp_ok = approx_eq(e0,e2) && approx_eq(e1,e3);
            if (!opp_ok) continue;

            float w = 0.5f*(e0+e2), h = 0.5f*(e1+e3);
            if (w < min_side_len_ || h < min_side_len_ || w > max_side_len_ || h > max_side_len_) continue;

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

  static bool intersectLinesST(const Line2D& A, const Line2D& B,
                               Eigen::Vector2f& out, float& s_out, float& t_out,
                               bool clamp_to_segments = true)
  {
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
    out = p0 + s*u; s_out=s; t_out=t; return true;
  }

  int countSupportOnSegment(const Line2D& L, float sa, float sb,
                            double /*tol_dist*/, int min_pts_required)
  {
    if (sa > sb) std::swap(sa, sb);
    float ia = std::max(sa, L.smin_raw);
    float ib = std::min(sb, L.smax_raw);
    if (ia > ib) return 0;
    int cnt = 0;
    for (float s : L.s_points){
      if (s < ia || s > ib) continue;
      if (++cnt >= min_pts_required) break;
    }
    return cnt;
  }

  void addCentersDedup(const std::vector<Eigen::Vector2f>& new_cs){
    for (const auto& c : new_cs){
      bool dup=false;
      for (const auto& e : centers_accum_){
        if ((e - c).norm() <= (float)center_dedupe_radius_){ dup=true; break; }
      }
      if (!dup) centers_accum_.push_back(c);
    }
  }

  // 🔁 equalize: “전체 점”과 별도로 “삽입된 점”을 반환
  struct EqualizeResult {
    std::vector<Eigen::Vector2f> all_points_sorted;
    std::vector<Eigen::Vector2f> inserted_only;  // 새로 만든 점들
    Eigen::Vector2f line_p0;
    Eigen::Vector2f line_u;
    double line_yaw{0.0};
  };

  EqualizeResult equalizeOnLine(const std::vector<Eigen::Vector2f>& pts)
  {
    EqualizeResult R;
    auto L = fitLinePCA_pts(pts, /*extension_len=*/0.0);
    if (!L) return R;
    R.line_p0 = L->p0; R.line_u = L->u;
    R.line_yaw = std::atan2(R.line_u.y(), R.line_u.x());

    struct SItem { float s; Eigen::Vector2f p; };
    std::vector<SItem> sitems; sitems.reserve(pts.size());
    for (auto& p : pts){
      float s = (p - L->p0).dot(L->u);
      sitems.push_back({s, p});
    }
    std::sort(sitems.begin(), sitems.end(),
              [](const SItem& a, const SItem& b){ return a.s < b.s; });

    if (sitems.empty()){
      return R;
    }
    if (sitems.size()==1){
      R.all_points_sorted.push_back(sitems.front().p);
      return R;
    }

    std::vector<float> gaps;
    for (size_t i=0;i+1<sitems.size();++i){
      float d = (sitems[i+1].p - sitems[i].p).norm();
      if (d > 1e-3f) gaps.push_back(d);
    }
    float base_gap = 0.f;
    if (!gaps.empty()){
      if (equalize_use_min_gap_) base_gap = *std::min_element(gaps.begin(), gaps.end());
      else {
        std::vector<float> tmp=gaps;
        std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/2, tmp.end());
        base_gap = tmp[tmp.size()/2];
      }
    } else {
      base_gap = 1.0f; // fallback
    }
    base_gap = std::max((float)base_gap_floor_, std::min((float)base_gap_ceil_, base_gap));

    // 첫 점
    R.all_points_sorted.push_back(sitems.front().p);

    for (size_t i=0;i+1<sitems.size();++i){
      float s0 = sitems[i].s;
      float s1 = sitems[i+1].s;
      float seg_len = std::fabs(s1 - s0);

      float k_real = seg_len / base_gap;
      int   k_int  = (int)std::round(k_real);
      if (k_int < 1) k_int = 1;

      if (std::fabs(k_real - (float)k_int) <= (float)spacing_insert_tol_){
        int n_insert = std::max(0, k_int - 1);
        for (int k=1; k<=n_insert; ++k){
          float s_new = s0 + (base_gap * k);
          Eigen::Vector2f p_new = L->p0 + s_new * L->u;
          R.all_points_sorted.push_back(p_new);
          R.inserted_only.push_back(p_new);                 // ✅ 삽입점 기록
        }
      } else {
        int n_insert = (int)std::round(seg_len / base_gap) - 1;
        if (n_insert < 0) n_insert = 0;
        for (int k=1; k<=n_insert; ++k){
          float s_new = s0 + (base_gap * k);
          Eigen::Vector2f p_new = L->p0 + s_new * L->u;
          R.all_points_sorted.push_back(p_new);
          R.inserted_only.push_back(p_new);                 // ✅ 삽입점 기록
        }
      }

      R.all_points_sorted.push_back(sitems[i+1].p); // 원래 점
    }

    // s 기준 재정렬
    std::sort(R.all_points_sorted.begin(), R.all_points_sorted.end(),
              [&](const Eigen::Vector2f& A, const Eigen::Vector2f& B){
                float sa = (A - L->p0).dot(L->u);
                float sb = (B - L->p0).dot(L->u);
                return sa < sb;
              });
    return R;
  }

  // ---------------- 메인 콜백 ----------------
  void onMarkers(const visualization_msgs::msg::MarkerArray::SharedPtr msg){
    if (!msg->markers.empty()) last_header_ = msg->markers.front().header;

    // 그룹 분리
    std::vector<Group> groupsX, groupsY;
    for (const auto& m : msg->markers){
      if (m.type != visualization_msgs::msg::Marker::POINTS) continue;
      if (m.action == visualization_msgs::msg::Marker::DELETE) continue;
      if (m.points.size() < 2) continue;
      Group g; g.ns=m.ns; g.id=m.id; g.pts.reserve(m.points.size());
      for (const auto& P : m.points) g.pts.emplace_back(P.x, P.y);
      if (m.ns=="groupX_pts") groupsX.push_back(std::move(g));
      else if (m.ns=="groupY_pts") groupsY.push_back(std::move(g));
    }

    // 선 핏팅(초록용)
    std::vector<Line2D> linesX, linesY;
    for (auto& g: groupsX){ if (auto L=fitLinePCA_pts(g.pts, extension_len_)) linesX.push_back(*L); }
    for (auto& g: groupsY){ if (auto L=fitLinePCA_pts(g.pts, extension_len_)) linesY.push_back(*L); }

    // 선 시각화
    publishLines(linesX, linesY);

    // 초록 사각형 생성 및 시각화
    auto green_rects = generateGreenRects(linesX, linesY);
    publishGreenRects(green_rects);

    // 1) 초록 사각형 중심 누적 (중복 제거)
    std::vector<Eigen::Vector2f> centers_now;
    centers_now.reserve(green_rects.size());
    for (auto& R: green_rects){
      Eigen::Vector2f c = 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
      centers_now.push_back(c);
    }
    addCentersDedup(centers_now);

    // 2) 유사 직선 생성 + 균등 간격 삽입점 생성
    if ((int)centers_accum_.size() >= min_pts_for_line_){
      auto eq = equalizeOnLine(centers_accum_);

      // 3) 삽입된 점들만 /parking_goal 로 PoseStamped 발행 + 시각화
      if (!eq.inserted_only.empty()){
        for (const auto& p : eq.inserted_only){
          geometry_msgs::msg::PoseStamped goal;
          goal.header.stamp = this->get_clock()->now();
          goal.header.frame_id = last_header_.frame_id;    // 맵 좌표(입력 마커 프레임)
          goal.pose.position.x = p.x();
          goal.pose.position.y = p.y();
          goal.pose.position.z = 0.0;
          goal.pose.orientation = yawToQuat(eq.line_yaw);
          pub_goal_->publish(goal);
        }
      }

      publishInsertedOnlyVis(eq.inserted_only, eq.line_p0, eq.line_u);
      RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 1000,
        "inserted goals: %zu (accum centers=%zu)", eq.inserted_only.size(), centers_accum_.size());
    }
  }

  // ---------------- 시각화 도우미 ----------------
  void publishLines(const std::vector<Line2D>& linesX, const std::vector<Line2D>& linesY){
    visualization_msgs::msg::MarkerArray line_arr;

    visualization_msgs::msg::Marker mx;
    mx.header=last_header_; mx.ns="rect_lines_x"; mx.id=0;
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
    my.header=last_header_; my.ns="rect_lines_y"; my.id=0;
    my.type=visualization_msgs::msg::Marker::LINE_LIST;
    my.action=visualization_msgs::msg::Marker::ADD;
    my.scale.x=0.06; my.color.r=0.1f; my.color.g=0.7f; my.color.b=1.0f; my.color.a=0.4f;
    for (auto& L: linesY){
      Eigen::Vector2f A=L.p0+L.s_min*L.u, B=L.p0+L.s_max*L.u;
      geometry_msgs::msg::Point pA,pB; pA.x=A.x(); pA.y=A.y(); pA.z=0; pB.x=B.x(); pB.y=B.y(); pB.z=0;
      my.points.push_back(pA); my.points.push_back(pB);
    }
    line_arr.markers.push_back(my);

    pub_lines_->publish(line_arr);
  }

  void publishGreenRects(const std::vector<Rect>& rects){
    visualization_msgs::msg::MarkerArray rect_arr;

    // edge
    {
      visualization_msgs::msg::Marker m;
      m.header=last_header_; m.ns="rects"; m.id=0;
      m.type=visualization_msgs::msg::Marker::LINE_LIST;
      m.scale.x=0.1;
      m.color.r=0.2f; m.color.g=1.0f; m.color.b=0.2f; m.color.a=0.98f;
      for (auto& R: rects){
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
    // label
    {
      int rid=0;
      for (auto& R: rects){
        visualization_msgs::msg::Marker t;
        t.header=last_header_; t.ns="rect_labels"; t.id=rid++;
        t.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action=visualization_msgs::msg::Marker::ADD;
        Eigen::Vector2f c = 0.25f*(R.c[0]+R.c[1]+R.c[2]+R.c[3]);
        t.pose.position.x=c.x(); t.pose.position.y=c.y(); t.pose.position.z=0.1;
        t.scale.z=0.35;
        t.color.r=1; t.color.g=1; t.color.b=1; t.color.a=1;
        char buf[64]; std::snprintf(buf,sizeof(buf),"w=%.2f,h=%.2f",R.w,R.h);
        t.text=buf;
        rect_arr.markers.push_back(t);
      }
      for (int i=(int)rects.size(); i<last_rect_label_count_; ++i){
        visualization_msgs::msg::Marker del;
        del.header=last_header_; del.ns="rect_labels"; del.id=i;
        del.action=visualization_msgs::msg::Marker::DELETE;
        rect_arr.markers.push_back(del);
      }
      last_rect_label_count_ = (int)rects.size();
      RCLCPP_INFO(get_logger(), "초록 사각형 개수: %zu", rects.size());
    }

    pub_rects_->publish(rect_arr);
  }

  // 🔁 “삽입된 점만” 시각화
  void publishInsertedOnlyVis(const std::vector<Eigen::Vector2f>& inserted,
                              const Eigen::Vector2f& p0,
                              const Eigen::Vector2f& u)
  {
    visualization_msgs::msg::MarkerArray arr;

    // 삽입점이 없으면 마커 삭제
    {
      visualization_msgs::msg::Marker del_pts;
      del_pts.header=last_header_; del_pts.ns="inserted_pts"; del_pts.id=0;
      del_pts.action = inserted.empty()? visualization_msgs::msg::Marker::DELETE
                                       : visualization_msgs::msg::Marker::ADD;
      del_pts.type=visualization_msgs::msg::Marker::SPHERE_LIST;
      del_pts.scale.x=0.2; del_pts.scale.y=0.2; del_pts.scale.z=0.2;
      del_pts.color.r=0.95f; del_pts.color.g=0.3f; del_pts.color.b=1.0f; del_pts.color.a=0.95f;
      for (auto& p : inserted){
        geometry_msgs::msg::Point gp; gp.x=p.x(); gp.y=p.y(); gp.z=0.0;
        del_pts.points.push_back(gp);
      }
      arr.markers.push_back(del_pts);
    }

    // 인덱스 라벨
    {
      int idx=0;
      for (auto& p : inserted){
        visualization_msgs::msg::Marker t;
        t.header=last_header_; t.ns="inserted_idx"; t.id=idx;
        t.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action=inserted.empty()? visualization_msgs::msg::Marker::DELETE
                                 : visualization_msgs::msg::Marker::ADD;
        t.pose.position.x=p.x(); t.pose.position.y=p.y(); t.pose.position.z=0.3;
        t.scale.z=0.30;
        t.color.r=1.0; t.color.g=1.0; t.color.b=1.0; t.color.a=1.0;
        t.text = std::to_string(idx);
        arr.markers.push_back(t);
        ++idx;
      }
    }

    // 직선은 참고용으로 양 끝만(삽입점이 있을 때만)
    if (!inserted.empty()){
      // 삽입점 s범위
      std::vector<float> svals; svals.reserve(inserted.size());
      for (auto& p : inserted) svals.push_back((p - p0).dot(u));
      auto [smin_it, smax_it] = std::minmax_element(svals.begin(), svals.end());
      float smin = *smin_it, smax = *smax_it;

      visualization_msgs::msg::Marker line;
      line.header=last_header_; line.ns="inserted_line"; line.id=0;
      line.type=visualization_msgs::msg::Marker::LINE_STRIP;
      line.action=visualization_msgs::msg::Marker::ADD;
      line.scale.x=0.06;
      line.color.r=0.2f; line.color.g=0.9f; line.color.b=0.9f; line.color.a=0.8f;
      Eigen::Vector2f A = p0 + smin * u;
      Eigen::Vector2f B = p0 + smax * u;
      geometry_msgs::msg::Point gA, gB; gA.x=A.x(); gA.y=A.y(); gA.z=0; gB.x=B.x(); gB.y=B.y(); gB.z=0;
      line.points.push_back(gA); line.points.push_back(gB);
      arr.markers.push_back(line);
    } else {
      visualization_msgs::msg::Marker del_line;
      del_line.header=last_header_; del_line.ns="inserted_line"; del_line.id=0;
      del_line.action=visualization_msgs::msg::Marker::DELETE;
      arr.markers.push_back(del_line);
    }

    pub_ins_vis_->publish(arr);
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ParkingPreNode>());
  rclcpp::shutdown();
  return 0;
}
