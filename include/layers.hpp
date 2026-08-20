#pragma once

#include "activations.hpp"
#include "matrix.hpp"

template <typename T>
class BaseLayer {
    public:
        virtual Matrix<T> backward(const Matrix<T>& gradient) = 0;
        virtual Matrix<T> forward(const Matrix<T>& input) = 0;      
        virtual void update(double learning_rate) = 0;
        virtual ~BaseLayer() = default;
};

template <typename T>
class ActivationLayer : public BaseLayer<T> {};

template <typename T>
class LinearLayer : public BaseLayer<T> {
    private:
        Matrix<T> last_input;
        Matrix<T> weights;
        Matrix<T> biases;

        Matrix<T> dLdW;
        Matrix<T> dLdb;

    public:
        LinearLayer(std::size_t input_size, std::size_t output_size) :
            weights(Matrix<T>::random(input_size, output_size)), 
            biases(Matrix<T>::random(1, output_size)) {}

        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;
            return Matrix<T>::multiply(input, weights) + biases;
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
            Matrix<T> dLdX = Matrix<T>::multiply(gradient, weights.transpose());
            Matrix<T> db(1, gradient.columns);

            for (std::size_t j = 0; j < gradient.columns; ++j) {
                T sum = static_cast<T>(0);

                for (std::size_t i = 0; i < gradient.rows; ++i) {
                    sum += gradient.get(i, j);
                }

                db.set(0, j, sum);
            }

            this->dLdW = Matrix<T>::multiply(last_input.transpose(), gradient);
            this->dLdb = db;

            return dLdX;
        }

        void update(double learning_rate) {
            T lr = static_cast<T>(learning_rate);

            weights = weights - (dLdW * lr);
            biases = biases - (dLdb * lr);
        }
};

template <typename T>
class LogSoftmax : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            Matrix<T> output(input.rows, input.columns);

            for (std::size_t i = 0; i < input.rows; ++i) {
                T max = input.get(i, 0);

                for (std::size_t j = 1; j < input.columns; ++j) {
                    if (input.get(i, j) > max) {
                        max = input.get(i, j);
                    }
                }

                T sum = static_cast<T>(0);
                for (std::size_t j = 0; j < input.columns; ++j) {
                    sum += std::exp(input.get(i, j) - max);
                }

                T log_sum = std::log(sum);

                for (std::size_t j = 0; j < input.columns; ++j) {
                    T value = (input.get(i, j) - max) - log_sum;
                    output.set(i, j, value);
                }
            }

            return output;
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
            return gradient;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class LeakyReLU : public ActivationLayer<T> {
    private:
        double alpha = 0.01;

    public:
        LeakyReLU(double alpha) : alpha(alpha) {}

        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;
            return input.apply([this](T x) {
                return leaky_relu<T>(x, alpha);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
           Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * leaky_relu_derivative<T>(x, alpha));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class Softmax : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;
            Matrix<T> output(input.rows, input.columns);

            for (std::size_t i = 0; i < input.rows; ++i) {
                T max = input.get(i, 0);

                for (std::size_t j = 1; j < input.columns; ++j) {
                    if (input.get(i, j) > max) {
                        max = input.get(i, j);
                    }
                }

                T sum = static_cast<T>(0);
                for (std::size_t j = 0; j < input.columns; ++j) {
                    sum += std::exp(input.get(i, j) - max);
                }

                for (std::size_t j = 0; j < input.columns; ++j) {
                    T value = std::exp(input.get(i, j) - max) / sum;
                    output.set(i, j, value);
                }
            }

            return output;
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
            return gradient;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class Sigmoid : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            return input.apply([](T x) {
                return sigmoid<T>(x);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
           Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * sigmoid_derivative<T>(x));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class Swish : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            return input.apply([](T x) {
                return swish<T>(x);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
           Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * swish_derivative<T>(x));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class ReLU : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            return input.apply([](T x) {
                return relu<T>(x);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
            Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * relu_derivative<T>(x));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class Tanh : public ActivationLayer<T> {
    public:
       Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            return input.apply([](T x) {
                return std::tanh(x);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
           Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * tanh_derivative<T>(x));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};

template <typename T>
class GELU : public ActivationLayer<T> {
    public:
        Matrix<T> forward(const Matrix<T>& input) override {
            last_input = input;

            return input.apply([](T x) {
                return gelu<T>(x);
            });
        }

        Matrix<T> backward(const Matrix<T>& gradient) override {
            Matrix<T> result(gradient.rows, gradient.columns);

            for (std::size_t i = 0; i < gradient.rows; ++i) {
                for (std::size_t j = 0; j < gradient.columns; ++j) {
                    T x = last_input.get(i, j);
                    T g = gradient.get(i, j);

                    result.set(i, j, g * gelu_derivative<T>(x));
                }
            }

            return result;
        }

        void update(double learning_rate) {}

    private:
        Matrix<T> last_input;
};