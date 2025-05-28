#include <cmath>
#include <queue>
#include <string>
#include <cassert>
#include <algorithm>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double axis_angle(Eigen::Matrix3d R1, Eigen::Matrix3d R2) {
    double trace = (R1 * R2.transpose()).trace();
    trace = (trace - 1) / 2;
    trace = trace < -1 ? -1: trace;
    trace = trace > 1 ? 1: trace;
    
    double theta = acos(trace) * 180 / M_PI;

    return theta;
}

double chordal_distance(Eigen::Matrix3d R1, Eigen::Matrix3d R2) {
    return (R1 - R2).norm();
}

double find_median(std::vector<double> array) { 
  
    int n = array.size();
    if (n % 2 == 0) { 
        nth_element(array.begin(), array.begin() + n / 2, array.end()); 
        nth_element(array.begin(), array.begin() + (n - 1) / 2, array.end()); 
  
        return (array[(n - 1) / 2] + array[n / 2]) / 2.0; 
    } else { 
        nth_element(array.begin(), array.begin() + n / 2, array.end()); 
  
        return array[n / 2]; 
    } 
} 

py::array_t<double> fit_error(
        py::array_t<long> adjacent_matrix, 
        int camera_num, 
        py::array_t<double> w_obs,
        py::array_t<double> w_fit,
        std::string metric
    ) {
    
    py::buffer_info adj_mat = adjacent_matrix.request();
    py::buffer_info obs_buffer = w_obs.request();
    py::buffer_info fit_buffer = w_fit.request();
    
    assert (adj_mat.ndim == 2 && obs_buffer.ndim == 2 && fit_buffer.ndim == 2);
    assert (adj_mat.shape[0] == adj_mat.shape[1] && obs_buffer.shape[0] == obs_buffer.shape[1] && fit_buffer.shape[0] == fit_buffer.shape[1]);
    assert (adj_mat.shape[0] == camera_num && obs_buffer.shape[0] == camera_num * 3 && fit_buffer.shape[0] == camera_num * 3);

    auto result = py::array_t<double>(adj_mat.size); 
    py::buffer_info result_mat = result.request();

    long* adj_mat_ptr = (long*)adj_mat.ptr;
    double* obs_ptr = (double*)obs_buffer.ptr;
    double* fit_ptr = (double*)fit_buffer.ptr;
    double* result_ptr = (double*)result_mat.ptr;

    for (int i = 0; i < camera_num; i++) {
        for (int j = 0; j < camera_num; j++) {
            
            if (i == j || adj_mat_ptr[i * camera_num + j] == 0) {
                result_ptr[i * camera_num + j] = 0;
                continue;
            }
            
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_obs, R_fit;
            for (int dx = 0; dx < 3; dx++) {
                for (int dy = 0; dy < 3; dy++) {
                    R_obs(dx, dy) = obs_ptr[(i * 3 + dx) * camera_num * 3 + j * 3 + dy];
                    R_fit(dx, dy) = fit_ptr[(i * 3 + dx) * camera_num * 3 + j * 3 + dy];
                }
            }

            double fit_error = 0;
            if (!metric.compare("chordal")) {
                fit_error = chordal_distance(R_obs, R_fit);
            } else if (!metric.compare("axis_angle")) {
                fit_error = axis_angle(R_obs, R_fit);
            } else {
                throw "Invalid metric!";
            }

            result_ptr[i * camera_num + j] = fit_error;
        }
    }
    
    return result;
}

class dsu {
private:
    std::vector<int> pa;
public:
    explicit dsu(int size_) : pa(size_) {
        iota(pa.begin(), pa.end(), 0);
    }

    void unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return;
        pa[y] = x;
    }

    int find(int x) { 
        return pa[x] == x ? x : pa[x] = find(pa[x]); 
    }
};

class edge {
public:
    int x, y;
    double weight;
    int support_count;
    explicit edge(int x_ = 0, int y_ = 0, double weight_ = 0.0, int support_count_ = 0) : x(x_), y(y_), weight(weight_), support_count(support_count_) {}

    void set(int x_, int y_, double weight_, int support_count_ = 0) {
        this->x = x_;
        this->y = y_;
        this->weight = weight_;
        this->support_count = support_count_;
    }

    bool operator<(const edge e) {
        if (support_count == e.support_count) {
            return weight < e.weight;
        }
        return support_count > e.support_count;
    }
};

py::array_t<long> support_spanning_tree(
        py::array_t<long> adjacent_matrix, 
        int camera_num, 
        py::array_t<double> w_obs,
        std::string metric
    ) {

    py::buffer_info adj_mat = adjacent_matrix.request();
    py::buffer_info w = w_obs.request();
    
    assert (adj_mat.ndim == 2 && w.ndim == 2);
    assert (adj_mat.shape[0] == adj_mat.shape[1] && w.shape[0] == w.shape[1]);
    assert (adj_mat.shape[0] == camera_num && w.shape[0] == camera_num * 3);

    auto result = py::array_t<double>(adj_mat.size); 
    py::buffer_info result_mat = result.request();

    long* adj_mat_ptr = (long*)adj_mat.ptr;
    double* w_ptr = (double*)w.ptr;
    double* result_ptr = (double*)result_mat.ptr;

    std::vector<double> all_triple_errors;
    std::vector<std::vector<std::vector<double> > > triple_errors(camera_num, std::vector<std::vector<double> >(camera_num, std::vector<double>(0)));

    for (int i = 0; i < camera_num; i++) {
        result_ptr[i * camera_num + i] = 0.0;
        for (int j = i + 1; j < camera_num; j++) {
            if (adj_mat_ptr[i * camera_num + j] == 0) {
                result_ptr[i * camera_num + j] = 1.0e9;
                result_ptr[j * camera_num + i] = 1.0e9;
                continue;
            }
            int loop_cnt = 0;
            result_ptr[i * camera_num + j] = 0;
            result_ptr[j * camera_num + i] = 0;
            for (int k = 0; k < camera_num; k++) {
                if (k == i || k == j || adj_mat_ptr[i * camera_num + k] == 0 || adj_mat_ptr[k * camera_num + j] == 0) {
                    continue;
                }

                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_ij, R_ik, R_kj;
                for (int dx = 0; dx < 3; dx++) {
                    for (int dy = 0; dy < 3; dy++) {
                        R_ij(dx, dy) = w_ptr[(i * 3 + dx) * camera_num * 3 + j * 3 + dy];
                        R_ik(dx, dy) = w_ptr[(i * 3 + dx) * camera_num * 3 + k * 3 + dy];
                        R_kj(dx, dy) = w_ptr[(k * 3 + dx) * camera_num * 3 + j * 3 + dy];
                    }
                }

                double loop_error = 0;
                if (!metric.compare("chordal")) {
                    loop_error = chordal_distance(R_ij, R_ik * R_kj);
                } else if (!metric.compare("axis_angle")) {
                    loop_error = axis_angle(R_ij, R_ik * R_kj);
                } else {
                    throw "Invalid metric!";
                }

                result_ptr[i * camera_num + j] += loop_error;
                result_ptr[j * camera_num + i] += loop_error;
                loop_cnt++;
                all_triple_errors.push_back(loop_error);
                triple_errors[i][j].push_back(loop_error);
            }
            if (loop_cnt > 0) {
                result_ptr[i * camera_num + j] /= loop_cnt;
                result_ptr[j * camera_num + i] /= loop_cnt;
            }
        }
    }

    double triple_median = find_median(all_triple_errors);
    
    std::vector<edge> edges;
    
    for (int i = 0; i < camera_num; i++) {
        for (int j = i + 1; j < camera_num; j++) {    
            if (adj_mat_ptr[i * camera_num + j] == 0) { 
                continue;
            }

            int support_cnt = 0;
            for (auto loop_error: triple_errors[i][j]) {
                if (loop_error < triple_median) {
                    support_cnt++;
                }
            }

            edges.emplace_back(i, j, result_ptr[i * camera_num + j], support_cnt);
        }
    }

    std::sort(edges.begin(), edges.end());

    dsu node(camera_num);

    auto tree = py::array_t<long>(adj_mat.size); 
    py::buffer_info tree_mat = tree.request();
    long* tree_mat_ptr = (long*)tree_mat.ptr;
    memset(tree_mat_ptr, 0, sizeof(long) * camera_num * camera_num);
    
    int temp = 0;
    for (auto graph_edge: edges) {
        int pa_x = node.find(graph_edge.x), pa_y = node.find(graph_edge.y); 
        if (pa_x == pa_y) {
            continue;
        }

        tree_mat_ptr[graph_edge.x * camera_num + graph_edge.y] = 1;
        tree_mat_ptr[graph_edge.y * camera_num + graph_edge.x] = 1;
        
        node.unite(graph_edge.x, graph_edge.y);
        
        temp++;
        if (temp == camera_num - 1) {
            break;
        }
    }

    return tree;
}

py::array_t<double> get_pose_from_tree(
        int camera_num,
        int root,
        py::array_t<long> tree_matrix,
        py::array_t<double> w_obs
    ) {

    py::buffer_info tree_mat = tree_matrix.request();
    py::buffer_info w = w_obs.request();
    
    assert (0 <= root && root < camera_num);
    
    assert (tree_mat.ndim == 2);
    assert (w.ndim == 2);

    assert (tree_mat.shape[0] == tree_mat.shape[1]);
    assert (w.shape[0] == w.shape[1]);

    assert (tree_mat.shape[0] == camera_num);
    assert (w.shape[0] == camera_num * 3);

    auto result = py::array_t<double>(camera_num * 3 * 3);
    py::buffer_info result_mat = result.request();

    long* tree_mat_ptr = (long*)tree_mat.ptr;
    double* w_ptr = (double*)w.ptr;
    double* result_ptr = (double*)result_mat.ptr;
    memset(result_ptr, 0, sizeof(double) * camera_num * 3 * 3);

    std::queue<int> q;
    bool* vis = new bool[camera_num];
    memset(vis, 0, sizeof(bool) * camera_num);
    
    q.push(root);
    result_ptr[root * 9] = result_ptr[root * 9 + 4] = result_ptr[root * 9 + 8] = 1;
    vis[root] = true;

    while (!q.empty()) {
        auto temp = q.front();
        q.pop();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> pos, pos_i;
        for (int dx = 0; dx < 3; dx++) {
            for (int dy = 0; dy < 3; dy++) {
                pos(dx, dy) = result_ptr[(temp * 3 + dx) * 3 + dy];
            }
        }

        for (int i = 0; i < camera_num; i++) {
            if (tree_mat_ptr[temp * camera_num + i] == 1 && !vis[i]) {
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation;
                for (int dx = 0; dx < 3; dx++) {
                    for (int dy = 0; dy < 3; dy++) {
                        rotation(dx, dy) = w_ptr[(temp * 3 + dx) * camera_num * 3 + i * 3 + dy];
                    }
                }

                pos_i = pos * rotation;

                for (int dx = 0; dx < 3; dx++) {
                    for (int dy = 0; dy < 3; dy++) {
                        result_ptr[(i * 3 + dx) * 3 + dy] = pos_i(dx, dy);
                    }
                }

                q.push(i);
                vis[i] = true;
            }
        }
    }

    return result;
}


PYBIND11_MODULE(graph_ops, m) {
    m.def("fit_error", &fit_error);
    m.def("get_pose_from_tree", &get_pose_from_tree);
    m.def("support_spanning_tree", &support_spanning_tree);
}