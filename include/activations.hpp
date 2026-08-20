#pragma once

#include <cmath>

template <typename T>
inline T sigmoid(T x) {
    return (static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)));
}

template <typename T>
inline T sigmoid_derivative(T x) {
    return sigmoid<T>(x) * (static_cast<T>(1) - sigmoid<T>(x));
}

template <typename T>
inline T tanh_derivative(T x) {
    T value = std::tanh(x);
    return static_cast<T>(1) - (value * value);
}

template <typename T>
inline T relu(T x) {
    return std::max(static_cast<T>(0), x);
}

template <typename T>
inline T relu_derivative(T x) {
    return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
}

template <typename T>
inline T leaky_relu(T x, double alpha) {
    return x > static_cast<T>(0) ? x : (alpha * x);
}

template <typename T>
inline T leaky_relu_derivative(T x, double alpha) {
    return x > static_cast<T>(0) ? static_cast<T>(1) : alpha;
}

template <typename T>
inline T swish(T x) {
    return x * sigmoid<T>(x);
}

template <typename T>
inline T swish_derivative(T x) {
    return sigmoid<T>(x) + (x * sigmoid<T>(x) * (static_cast<T>(1) - sigmoid<T>(x)));
}

template <typename T>
inline T gelu(T x) {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T coefficient = static_cast<T>(0.044715);
    const T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);

    return half * x * (one + std::tanh(sqrt_2_over_pi * (x + coefficient * x * x * x)));
}

template <typename T>
inline T gelu_derivative(T x) {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T three = static_cast<T>(3);
    const T coefficient = static_cast<T>(0.044715);
    const T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);

    T x3 = x * x * x;
    T inner = sqrt_2_over_pi * (x + coefficient * x3);
    T tanh_inner = std::tanh(inner);

    T sech2 = one - (tanh_inner * tanh_inner);

    return half * (one + tanh_inner) + half * x * sech2 * sqrt_2_over_pi * (one + three * coefficient * x * x);
}