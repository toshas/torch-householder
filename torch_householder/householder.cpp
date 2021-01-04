#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <vector>

using namespace torch::indexing;


at::Tensor orgqr_normalized_eye_forward(torch::Tensor param, torch::Tensor eye) {
    TORCH_CHECK(param.is_floating_point(), "\"param\" must be a floating point tensor");
    TORCH_CHECK(param.dim() == 3, "\"param\" must have 3 dimensions");
    TORCH_CHECK(eye.dim() == 3, "\"eye\" must have 3 dimensions");
    TORCH_CHECK(!eye.requires_grad(), "\"eye\" must not require grad");
    auto b = param.size(0);
    auto d = param.size(1);
    auto r = param.size(2);
    TORCH_CHECK(eye.size(0) == b && eye.size(1) == d && eye.size(2) == r, "\"eye\" dimensions must match parameters")
    TORCH_CHECK(d >= r, "\"param\" dim[1] must be greater or equal than dim[2]");
    TORCH_CHECK(param.device() == eye.device(), "Device mismatch");
    const auto options = at::TensorOptions()
        .dtype(param.dtype())
        .device(param.device())
        .requires_grad(false);
    auto m = eye;
    auto minus_tau_param = -2 * param;
    auto ax = torch::empty_like(m);
    auto bx = torch::empty_like(m);
    auto cx = torch::empty({b, 1, r}, options);
    for (int j=r-1; j>=0; j--) {
        if (j < r-1) {
            m = ax;
        }
        auto uT = param.index({Slice(), Slice(), j}).unsqueeze(1);
        auto mtu = minus_tau_param.index({Slice(), Slice(), j}).unsqueeze(2);
        at::bmm_out(cx, uT, m);
        at::baddbmm_out(bx, m, mtu, cx);
        std::swap(ax, bx);
    }
    return ax;
}


at::Tensor orgqr_normalized_forward(torch::Tensor param) {
    TORCH_CHECK(param.is_floating_point(), "\"param\" must be a floating point tensor");
    TORCH_CHECK(param.dim() == 3, "\"param\" must have 3 dimensions");
    auto b = param.size(0);
    auto d = param.size(1);
    auto r = param.size(2);
    TORCH_CHECK(d >= r, "\"param\" dim[1] must be greater or equal than dim[2]");
    const auto options = at::TensorOptions()
        .dtype(param.dtype())
        .device(param.device())
        .requires_grad(false);
    auto eye = torch::eye(d, r, options).unsqueeze(0).repeat({b, 1, 1});
    return orgqr_normalized_eye_forward(param, eye);
}


at::Tensor orgqr_normalized_backward(torch::Tensor grad_out, torch::Tensor m, torch::Tensor param) {
    auto b = param.size(0);
    auto d = param.size(1);
    auto r = param.size(2);
    const auto options = at::TensorOptions()
        .dtype(param.dtype())
        .device(param.device())
        .requires_grad(false);
    auto grad_m = grad_out;
    auto grad_param = torch::empty_like(param);
    auto minus_tau_param = -2 * param;
    auto ax = torch::empty({b, 1, r}, options);
    auto bx = torch::empty({b, 1, r}, options);
    auto cx = torch::empty({b, d, 1}, options);
    auto dx = torch::empty({b, 1, 1}, options);
    auto gx = torch::empty_like(m);
    auto hx = torch::empty_like(m);
    auto mx = torch::empty_like(m);
    auto nx = torch::empty_like(m);
    for (int j=0; j<r; j++) {
        if (j > 0) {
            grad_m = gx;
            m = mx;
        }
        auto uT = param.index({Slice(), Slice(), j}).unsqueeze(1);
        auto mtu = minus_tau_param.index({Slice(), Slice(), j}).unsqueeze(2);
        torch::bmm_out(ax, uT, m);
        torch::bmm_out(bx, uT, grad_m);
        auto bxT = bx.permute({0, 2, 1});
        torch::bmm_out(dx, ax, bxT);
        torch::bmm_out(cx, m, bxT);
        torch::baddbmm_out(cx, cx, grad_m, ax.permute({0, 2, 1}), /*beta=*/1, /*alpha=*/-1);
        auto out = grad_param.index({Slice(), Slice(), j}).unsqueeze(2);
        torch::baddbmm_out(out, cx, mtu, dx, /*beta=*/-2, /*alpha=*/-2);
        if (j < r - 1) {
            torch::baddbmm_out(hx, grad_m, mtu, bx);
            torch::baddbmm_out(nx, m, mtu, ax);
            std::swap(gx, hx);
            std::swap(mx, nx);
        }
    }
    return grad_param;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("orgqr_normalized_forward", &orgqr_normalized_forward, "ORGQR normalized input forward");
    m.def("orgqr_normalized_eye_forward", &orgqr_normalized_eye_forward, "ORGQR normalized input with eye forward");
    m.def("orgqr_normalized_backward", &orgqr_normalized_backward, "ORGQR normalized input backward");
}
