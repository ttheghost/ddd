/**
 * @file tensor.hpp
 * @brief N-dimensional tensor.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dl {
  template<typename T>
    requires std::is_arithmetic_v<T>
  class Tensor {
    std::vector<size_t> shape_;
    size_t size_ = 0;
    std::vector<size_t> strides_;
    std::shared_ptr<T[]> data_;
    bool contiguous_ = true;

    // Internal helpers

    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape) {
      std::vector<size_t> s(shape.size(), 1);
      if (shape.size() <= 1) return s;
      for (size_t i = shape.size(); i-- > 1;)
        s[i - 1] = s[i] * shape[i];
      return s;
    }

    static size_t calculate_size(const std::vector<size_t> &shape) {
      if (shape.empty()) return 0;
      return std::accumulate(shape.begin(), shape.end(),
                             size_t{1}, std::multiplies<size_t>{});
    }

    static std::vector<size_t> broadcast_shapes(const std::vector<size_t> &a,
                                                const std::vector<size_t> &b) {
      const size_t nd = std::max(a.size(), b.size());
      std::vector<size_t> result(nd);
      for (size_t i = 0; i < nd; ++i) {
        size_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1)
          throw std::invalid_argument("Incompatible shapes for broadcasting");
        result[nd - 1 - i] = std::max(da, db);
      }
      return result;
    }

    static std::vector<size_t> get_broadcast_strides(const std::vector<size_t> &shape,
                                                     const std::vector<size_t> &strides,
                                                     const std::vector<size_t> &result_shape) {
      const size_t ndim_r = result_shape.size();
      const size_t ndim = shape.size();
      std::vector<size_t> eff(ndim_r, 0);
      const size_t off = ndim_r - ndim;
      for (size_t i = 0; i < ndim_r; ++i) {
        const size_t orig_dim = (i < off) ? 1 : shape[i - off];
        eff[i] = (orig_dim > 1) ? strides[i - off] : 0;
      }
      return eff;
    }

    /**
     * Map a logical flat index to a physical data offset using per-dimension
     * strides.  Iterates FIRST→LAST (highest stride first — correctness fix).
     * Called ONLY for non-contiguous tensors; hot paths bypass this.
     */
    static size_t flat_to_offset(size_t flat,
                                 const std::vector<size_t> &result_shape,
                                 const std::vector<size_t> &result_strides,
                                 const std::vector<size_t> &eff_strides) {
      size_t offset = 0;
      size_t temp = flat;
      const size_t nd = result_shape.size();
      for (size_t d = 0; d < nd; ++d) {
        const size_t idx = temp / result_strides[d];
        temp %= result_strides[d];
        offset += idx * eff_strides[d];
      }
      return offset;
    }

    [[nodiscard]] size_t flat_index(const std::initializer_list<size_t> &idx) const {
      if (idx.size() != shape_.size())
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
      size_t flat = 0, dim = 0;
      for (size_t i: idx) {
        if (i >= shape_[dim])
          throw std::out_of_range("Index out of bounds in dim " + std::to_string(dim));
        flat += i * strides_[dim++];
      }
      return flat;
    }

    /**
     * Read the i-th logical element.
     * HOT PATH: contiguous tensors short-circuit, no division, no allocation.
     */
    [[nodiscard]] T read_flat(size_t i) const {
      if (contiguous_) return data_[i];
      const auto ss = compute_strides(shape_);
      return data_[flat_to_offset(i, shape_, ss, strides_)];
    }

    // Private ctor for transpose / reshape views
    Tensor(std::vector<size_t> shape, std::vector<size_t> strides,
           std::shared_ptr<T[]> data, bool contiguous)
      : shape_(std::move(shape)), size_(calculate_size(shape_)),
        strides_(std::move(strides)), data_(std::move(data)),
        contiguous_(contiguous) {
    }

  public:
    // Constructors

    explicit Tensor(const std::vector<size_t> &shape)
      : shape_(shape), size_(calculate_size(shape_)),
        strides_(compute_strides(shape_)),
        data_(std::make_shared<T[]>(size_)),
        contiguous_(true) { fill(T{0}); }

    Tensor(const std::initializer_list<size_t> &shape)
      : shape_(shape), size_(calculate_size(shape_)),
        strides_(compute_strides(shape_)),
        data_(std::make_shared<T[]>(size_)),
        contiguous_(true) { fill(T{0}); }

    Tensor(const Tensor &) = default;

    Tensor &operator=(const Tensor &) = default;

    Tensor(Tensor &&o) noexcept
      : shape_(std::move(o.shape_)), size_(std::exchange(o.size_, 0)),
        strides_(std::move(o.strides_)), data_(std::move(o.data_)),
        contiguous_(o.contiguous_) {
    }

    Tensor &operator=(Tensor &&o) noexcept {
      if (this != &o) {
        shape_ = std::move(o.shape_);
        size_ = std::exchange(o.size_, 0);
        strides_ = std::move(o.strides_);
        data_ = std::move(o.data_);
        contiguous_ = o.contiguous_;
      }
      return *this;
    }

    // Basic ops

    [[nodiscard]] Tensor clone() const {
      Tensor out(shape_);
      if (contiguous_) {
        std::copy_n(data_.get(), size_, out.data_.get());
      } else {
        for (size_t i = 0; i < size_; ++i) out.data_[i] = read_flat(i);
      }
      return out;
    }

    void fill(T v) { std::fill_n(data_.get(), size_, v); }

    // Accessors

    [[nodiscard]] const std::vector<size_t> &shape() const { return shape_; }
    [[nodiscard]] const std::vector<size_t> &strides() const { return strides_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t ndim() const { return shape_.size(); }
    [[nodiscard]] bool is_contiguous() const { return contiguous_; }
    [[nodiscard]] T *data() { return data_.get(); }
    [[nodiscard]] const T *data() const { return data_.get(); }

    T &at(const std::initializer_list<size_t> &idx) { return data_[flat_index(idx)]; }
    const T &at(const std::initializer_list<size_t> &idx) const { return data_[flat_index(idx)]; }

    /**
     * Flat element access — FAST PATH for contiguous tensors.
     * Contiguous: data_[i] IS the i-th element (zero extra work).
     * Non-contiguous: stride decomposition (only for transposed views).
     */
    T &operator[](size_t i) {
      if (contiguous_) return data_[i]; // ← fast path
      if (i >= size_) throw std::out_of_range("Index out of bounds");
      const auto ss = compute_strides(shape_);
      return data_[flat_to_offset(i, shape_, ss, strides_)];
    }

    const T &operator[](size_t i) const {
      if (contiguous_) return data_[i];
      if (i >= size_) throw std::out_of_range("Index out of bounds");
      const auto ss = compute_strides(shape_);
      return data_[flat_to_offset(i, shape_, ss, strides_)];
    }

    // Shape manipulation

    [[nodiscard]] Tensor contiguous() const {
      if (contiguous_) return *this;
      return clone();
    }

    [[nodiscard]] Tensor reshape(const std::vector<size_t> &new_shape) const {
      if (calculate_size(new_shape) != size_)
        throw std::invalid_argument("reshape: element count mismatch");
      if (!contiguous_) return contiguous().reshape(new_shape);
      return Tensor(new_shape, compute_strides(new_shape), data_, true);
    }

    /**
     * Swap two axes.  Returns a non-contiguous VIEW (shares data_).
     */
    [[nodiscard]] Tensor transpose(const int dim0 = -2, const int dim1 = -1) const {
      if (ndim() < 2)
        throw std::invalid_argument("transpose requires at least 2 dimensions");
      const int d0 = (dim0 >= 0) ? dim0 : static_cast<int>(ndim()) + dim0;
      const int d1 = (dim1 >= 0) ? dim1 : static_cast<int>(ndim()) + dim1;
      if (d0 < 0 || d0 >= static_cast<int>(ndim()) ||
          d1 < 0 || d1 >= static_cast<int>(ndim()))
        throw std::out_of_range("transpose: dimension out of range");
      auto ns = shape_;
      auto nt = strides_;
      std::swap(ns[d0], ns[d1]);
      std::swap(nt[d0], nt[d1]);
      const bool is_cont = (d0 == d1); // only if no-op swap
      return Tensor(std::move(ns), std::move(nt), data_, is_cont);
    }

    [[nodiscard]] Tensor slice(int axis, size_t start, size_t end) const {
      const int nd = static_cast<int>(ndim());
      if (axis < 0) axis += nd;
      if (axis < 0 || axis >= nd) throw std::out_of_range("slice: axis out of range");
      if (end > shape_[axis] || start >= end)
        throw std::invalid_argument("slice: invalid start/end");

      std::vector<size_t> new_shape = shape_;
      new_shape[axis] = end - start;
      Tensor result(new_shape);

      const auto src_ss = compute_strides(shape_);
      const auto res_ss = compute_strides(new_shape);

      for (size_t flat = 0; flat < result.size_; ++flat) {
        std::vector<size_t> midx(nd);
        size_t tmp = flat;
        for (int d = 0; d < nd; ++d) {
          midx[d] = tmp / res_ss[d];
          tmp %= res_ss[d];
        }
        midx[axis] += start;
        size_t src_flat = 0;
        for (int d = 0; d < nd; ++d) src_flat += midx[d] * src_ss[d];
        result.data_[flat] = (contiguous_)
                               ? data_[src_flat]
                               : data_[flat_to_offset(src_flat, shape_, src_ss, strides_)];
      }
      return result;
    }

    // Reductions

    [[nodiscard]] T sum() const {
      T r{0};
      if (contiguous_) {
        for (size_t i = 0; i < size_; ++i) r += data_[i];
      } else {
        for (size_t i = 0; i < size_; ++i) r += read_flat(i);
      }
      return r;
    }

    [[nodiscard]] Tensor sum(int axis, bool keepdim = false) const {
      const int nd = static_cast<int>(ndim());
      if (axis < 0) axis += nd;
      if (axis < 0 || axis >= nd) throw std::out_of_range("sum: axis out of range");

      std::vector<size_t> res_shape;
      res_shape.reserve(nd);
      for (int d = 0; d < nd; ++d) {
        if (d == axis) { if (keepdim) res_shape.push_back(1); } else res_shape.push_back(shape_[d]);
      }
      if (res_shape.empty()) res_shape = {1};

      Tensor result(res_shape);
      result.fill(T{0});

      const auto src_ss = compute_strides(shape_);
      const auto res_ss = compute_strides(res_shape);

      for (size_t flat = 0; flat < size_; ++flat) {
        size_t tmp = flat, res_flat = 0, rd = 0;
        for (int d = 0; d < nd; ++d) {
          size_t idx_d = tmp / src_ss[d];
          tmp %= src_ss[d];
          if (d == axis) { if (keepdim) ++rd; } else res_flat += idx_d * res_ss[rd++];
        }
        result.data_[res_flat] += read_flat(flat);
      }
      return result;
    }

    [[nodiscard]] T mean() const {
      if (size_ == 0) throw std::invalid_argument("mean on empty tensor");
      return sum() / static_cast<T>(size_);
    }

    /** Global max — raw loop for contiguous. */
    [[nodiscard]] T max() const {
      if (size_ == 0) throw std::invalid_argument("max on empty tensor");
      if (contiguous_) {
        T best = data_[0];
        for (size_t i = 1; i < size_; ++i) if (data_[i] > best) best = data_[i];
        return best;
      }
      T best = read_flat(0);
      for (size_t i = 1; i < size_; ++i) {
        T v = read_flat(i);
        if (v > best) best = v;
      }
      return best;
    }

    /** Index of max element. */
    [[nodiscard]] size_t argmax() const {
      if (size_ == 0) throw std::invalid_argument("argmax on empty tensor");
      if (contiguous_) {
        size_t bi = 0;
        T bv = data_[0];
        for (size_t i = 1; i < size_; ++i)
          if (data_[i] > bv) {
            bv = data_[i];
            bi = i;
          }
        return bi;
      }
      size_t bi = 0;
      T bv = read_flat(0);
      for (size_t i = 1; i < size_; ++i) {
        T v = read_flat(i);
        if (v > bv) {
          bv = v;
          bi = i;
        }
      }
      return bi;
    }

    // Unary element-wise ops

    Tensor operator-() const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = -data_[i];
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = -read_flat(i);
      return r;
    }

    [[nodiscard]] Tensor exp() const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::exp(data_[i]);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::exp(read_flat(i));
      return r;
    }

    [[nodiscard]] Tensor log() const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::log(data_[i]);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::log(read_flat(i));
      return r;
    }

    [[nodiscard]] Tensor tanh() const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::tanh(data_[i]);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::tanh(read_flat(i));
      return r;
    }

    [[nodiscard]] Tensor sqrt() const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::sqrt(data_[i]);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::sqrt(read_flat(i));
      return r;
    }

    [[nodiscard]] Tensor pow(T exponent) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::pow(data_[i], exponent);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::pow(read_flat(i), exponent);
      return r;
    }

    [[nodiscard]] Tensor clamp(T lo, T hi) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = std::clamp(data_[i], lo, hi);
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = std::clamp(read_flat(i), lo, hi);
      return r;
    }

    // Tensor Op Tensor (broadcast-aware)
    //
    // Common case: same shape, both contiguous: raw parallel pointer loop.

    Tensor operator+(const Tensor &o) const {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        Tensor r(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
      }
      auto rs = broadcast_shapes(shape_, o.shape_);
      auto rss = compute_strides(rs);
      auto ea = get_broadcast_strides(shape_, strides_, rs);
      auto eb = get_broadcast_strides(o.shape_, o.strides_, rs);
      Tensor result(rs);
      for (size_t i = 0; i < result.size_; ++i)
        result.data_[i] = data_[flat_to_offset(i, rs, rss, ea)]
                          + o.data_[flat_to_offset(i, rs, rss, eb)];
      return result;
    }

    Tensor &operator+=(const Tensor &o) {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        for (size_t i = 0; i < size_; ++i) data_[i] += o.data_[i];
        return *this;
      }
      if (shape_ != o.shape_) throw std::invalid_argument("operator+=: shape mismatch");
      const auto ss = compute_strides(shape_);
      const auto ebo = get_broadcast_strides(o.shape_, o.strides_, shape_);
      for (size_t i = 0; i < size_; ++i)
        data_[flat_to_offset(i, shape_, ss, strides_)] += o.data_[flat_to_offset(i, shape_, ss, ebo)];
      return *this;
    }

    Tensor operator-(const Tensor &o) const {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        Tensor r(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
      }
      auto rs = broadcast_shapes(shape_, o.shape_);
      auto rss = compute_strides(rs);
      auto ea = get_broadcast_strides(shape_, strides_, rs);
      auto eb = get_broadcast_strides(o.shape_, o.strides_, rs);
      Tensor result(rs);
      for (size_t i = 0; i < result.size_; ++i)
        result.data_[i] = data_[flat_to_offset(i, rs, rss, ea)]
                          - o.data_[flat_to_offset(i, rs, rss, eb)];
      return result;
    }

    Tensor &operator-=(const Tensor &o) {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        for (size_t i = 0; i < size_; ++i) data_[i] -= o.data_[i];
        return *this;
      }
      if (shape_ != o.shape_) throw std::invalid_argument("operator-=: shape mismatch");
      const auto ss = compute_strides(shape_);
      const auto ebo = get_broadcast_strides(o.shape_, o.strides_, shape_);
      for (size_t i = 0; i < size_; ++i)
        data_[flat_to_offset(i, shape_, ss, strides_)] -= o.data_[flat_to_offset(i, shape_, ss, ebo)];
      return *this;
    }

    Tensor operator*(const Tensor &o) const {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        Tensor r(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] * o.data_[i];
        return r;
      }
      auto rs = broadcast_shapes(shape_, o.shape_);
      auto rss = compute_strides(rs);
      auto ea = get_broadcast_strides(shape_, strides_, rs);
      auto eb = get_broadcast_strides(o.shape_, o.strides_, rs);
      Tensor result(rs);
      for (size_t i = 0; i < result.size_; ++i)
        result.data_[i] = data_[flat_to_offset(i, rs, rss, ea)]
                          * o.data_[flat_to_offset(i, rs, rss, eb)];
      return result;
    }

    Tensor &operator*=(const Tensor &o) {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        for (size_t i = 0; i < size_; ++i) data_[i] *= o.data_[i];
        return *this;
      }
      if (shape_ != o.shape_) throw std::invalid_argument("operator*=: shape mismatch");
      const auto ss = compute_strides(shape_);
      const auto ebo = get_broadcast_strides(o.shape_, o.strides_, shape_);
      for (size_t i = 0; i < size_; ++i)
        data_[flat_to_offset(i, shape_, ss, strides_)] *= o.data_[flat_to_offset(i, shape_, ss, ebo)];
      return *this;
    }

    Tensor operator/(const Tensor &o) const {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        Tensor r(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] / o.data_[i];
        return r;
      }
      auto rs = broadcast_shapes(shape_, o.shape_);
      auto rss = compute_strides(rs);
      auto ea = get_broadcast_strides(shape_, strides_, rs);
      auto eb = get_broadcast_strides(o.shape_, o.strides_, rs);
      Tensor result(rs);
      for (size_t i = 0; i < result.size_; ++i)
        result.data_[i] = data_[flat_to_offset(i, rs, rss, ea)]
                          / o.data_[flat_to_offset(i, rs, rss, eb)];
      return result;
    }

    Tensor &operator/=(const Tensor &o) {
      if (shape_ == o.shape_ && contiguous_ && o.contiguous_) {
        for (size_t i = 0; i < size_; ++i) data_[i] /= o.data_[i];
        return *this;
      }
      if (shape_ != o.shape_) throw std::invalid_argument("operator/=: shape mismatch");
      const auto ss = compute_strides(shape_);
      const auto ebo = get_broadcast_strides(o.shape_, o.strides_, shape_);
      for (size_t i = 0; i < size_; ++i)
        data_[flat_to_offset(i, shape_, ss, strides_)] /= o.data_[flat_to_offset(i, shape_, ss, ebo)];
      return *this;
    }

    // Tensor Op Scalar

    Tensor operator+(T s) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] + s;
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = read_flat(i) + s;
      return r;
    }

    Tensor operator-(T s) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] - s;
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = read_flat(i) - s;
      return r;
    }

    Tensor operator*(T s) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] * s;
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = read_flat(i) * s;
      return r;
    }

    Tensor operator/(T s) const {
      Tensor r(shape_);
      if (contiguous_) for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] / s;
      else for (size_t i = 0; i < size_; ++i) r.data_[i] = read_flat(i) / s;
      return r;
    }

    Tensor &operator+=(T s) {
      if (contiguous_) for (size_t i = 0; i < size_; ++i) data_[i] += s;
      else for (size_t i = 0; i < size_; ++i) (*this)[i] += s;
      return *this;
    }

    Tensor &operator-=(T s) {
      if (contiguous_) for (size_t i = 0; i < size_; ++i) data_[i] -= s;
      else for (size_t i = 0; i < size_; ++i) (*this)[i] -= s;
      return *this;
    }

    Tensor &operator*=(T s) {
      if (contiguous_) for (size_t i = 0; i < size_; ++i) data_[i] *= s;
      else for (size_t i = 0; i < size_; ++i) (*this)[i] *= s;
      return *this;
    }

    Tensor &operator/=(T s) {
      if (contiguous_) for (size_t i = 0; i < size_; ++i) data_[i] /= s;
      else for (size_t i = 0; i < size_; ++i) (*this)[i] /= s;
      return *this;
    }

    friend Tensor operator+(T s, const Tensor &t) { return t + s; }

    friend Tensor operator-(T s, const Tensor &t) {
      Tensor r(t.shape_);
      if (t.contiguous_) for (size_t i = 0; i < t.size_; ++i) r.data_[i] = s - t.data_[i];
      else for (size_t i = 0; i < t.size_; ++i) r.data_[i] = s - t.read_flat(i);
      return r;
    }

    friend Tensor operator*(T s, const Tensor &t) { return t * s; }

    friend Tensor operator/(T s, const Tensor &t) {
      Tensor r(t.shape_);
      if (t.contiguous_) for (size_t i = 0; i < t.size_; ++i) r.data_[i] = s / t.data_[i];
      else for (size_t i = 0; i < t.size_; ++i) r.data_[i] = s / t.read_flat(i);
      return r;
    }

    // Vector ops

    [[nodiscard]] bool is_vector() const { return ndim() == 1; }

    T dot(const Tensor &other) const {
      if (size_ != other.size_) throw std::invalid_argument("dot: size mismatch");
      T r{0};
      if (contiguous_ && other.contiguous_) {
        for (size_t i = 0; i < size_; ++i) r += data_[i] * other.data_[i];
      } else {
        for (size_t i = 0; i < size_; ++i) r += (*this)[i] * other[i];
      }
      return r;
    }

    /**
     * Outer product: result[i,j] = u[i] * v[j].
     * HOT PATH: both u and v are always contiguous column vectors in our use.
     * Raw pointer loop: no operator[], no initializer_list overhead.
     */
    [[nodiscard]] static Tensor outer(const Tensor &u, const Tensor &v) {
      const size_t m = u.size_, n = v.size_;
      Tensor result({m, n});
      T *r = result.data_.get();
      const T *pu = u.data_.get();
      const T *pv = v.data_.get();
      if (u.contiguous_ && v.contiguous_) {
        for (size_t i = 0; i < m; ++i) {
          const T ui = pu[i];
          T *ri = r + i * n;
          for (size_t j = 0; j < n; ++j) ri[j] = ui * pv[j];
        }
      } else {
        for (size_t i = 0; i < m; ++i) {
          const T ui = u[i];
          T *ri = r + i * n;
          for (size_t j = 0; j < n; ++j) ri[j] = ui * v[j];
        }
      }
      return result;
    }

    // Matrix ops

    [[nodiscard]] bool is_matrix() const { return ndim() == 2; }

    /**
     * Matrix multiplication — optimized for the common network cases:
     *
     *   Case A (forward):  contiguous W(M,K) × contiguous x(K,N)
     *     A[i,k] = a[i*K + k],  B[k,j] = b[k*N + j]
     *     result[i,j] = r[i*N + j]   — all raw pointer arithmetic, no overhead.
     *
     *   Case B (backward): transposed W^T(K,M) × contiguous grad(M,N)
     *     A[i,k] = a[i*sa0 + k*sa1]  with sa0,sa1 from the transposed strides
     *     B[k,j] = b[k*N + j]  (contiguous)
     *     result is always contiguous.
     */
    [[nodiscard]] Tensor matmul(const Tensor &other) const {
      if (!(is_matrix() && other.is_matrix()))
        throw std::invalid_argument("matmul: both operands must be 2-D");
      const size_t M = shape_[0], K = shape_[1], N = other.shape_[1];
      if (K != other.shape_[0])
        throw std::invalid_argument("matmul: inner dimensions mismatch");

      Tensor result({M, N}); // always contiguous, filled with 0
      T *r = result.data_.get();
      const T *a = data_.get();
      const T *b = other.data_.get();

      if (contiguous_ && other.contiguous_) {
        // both contiguous
        // ikj loop order — better cache utilization for the (K,1) column case
        for (size_t i = 0; i < M; ++i) {
          const T *ai = a + i * K;
          T *ri = r + i * N;
          for (size_t k = 0; k < K; ++k) {
            const T aik = ai[k];
            const T *bk = b + k * N;
            for (size_t j = 0; j < N; ++j) ri[j] += aik * bk[j];
          }
        }
      } else {
        // use stored strides (handles transpose)
        const size_t sa0 = strides_[0], sa1 = strides_[1];
        const size_t sb0 = other.strides_[0], sb1 = other.strides_[1];
        for (size_t i = 0; i < M; ++i) {
          T *ri = r + i * N;
          for (size_t k = 0; k < K; ++k) {
            const T aik = a[i * sa0 + k * sa1];
            for (size_t j = 0; j < N; ++j)
              ri[j] += aik * b[k * sb0 + j * sb1];
          }
        }
      }
      return result;
    }

    // Concatenation

    [[nodiscard]] static Tensor concat(const Tensor &a, const Tensor &b, int axis) {
      if (a.ndim() != b.ndim()) throw std::invalid_argument("concat: ndim mismatch");
      const int nd = static_cast<int>(a.ndim());
      if (axis < 0) axis += nd;
      for (int d = 0; d < nd; ++d)
        if (d != axis && a.shape_[d] != b.shape_[d])
          throw std::invalid_argument("concat: non-axis shapes differ");

      std::vector<size_t> rs = a.shape_;
      rs[axis] += b.shape_[axis];
      Tensor result(rs);

      const auto res_ss = compute_strides(rs);
      const auto a_ss = compute_strides(a.shape_);
      const auto b_ss = compute_strides(b.shape_);

      for (size_t flat = 0; flat < result.size_; ++flat) {
        std::vector<size_t> midx(nd);
        size_t tmp = flat;
        for (int d = 0; d < nd; ++d) {
          midx[d] = tmp / res_ss[d];
          tmp %= res_ss[d];
        }

        if (midx[axis] < a.shape_[axis]) {
          size_t af = 0;
          for (int d = 0; d < nd; ++d) af += midx[d] * a_ss[d];
          result.data_[flat] = a.contiguous_
                                 ? a.data_[af]
                                 : a.data_[flat_to_offset(af, a.shape_, a_ss, a.strides_)];
        } else {
          std::vector<size_t> bm = midx;
          bm[axis] -= a.shape_[axis];
          size_t bf = 0;
          for (int d = 0; d < nd; ++d) bf += bm[d] * b_ss[d];
          result.data_[flat] = b.contiguous_
                                 ? b.data_[bf]
                                 : b.data_[flat_to_offset(bf, b.shape_, b_ss, b.strides_)];
        }
      }
      return result;
    }

    // Debug

    [[nodiscard]] std::string to_string() const {
      std::ostringstream ss;
      ss << "[";
      for (size_t i = 0; i < size_; ++i)
        ss << read_flat(i) << (i < size_ - 1 ? ", " : "");
      ss << "]";
      return ss.str();
    }
  };
} // namespace dl
