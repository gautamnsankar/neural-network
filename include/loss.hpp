#pragma once

#include <cmath>

#include "matrix.hpp"

template <typename T>
class BaseLossFunction {
    public:
        virtual Matrix<T> gradient(const Matrix<T>& prediction, const Matrix<T>& target) = 0;
        virtual T loss(const Matrix<T>& prediction, const Matrix<T>& target) = 0;
        virtual ~BaseLossFunction() = default;
};

template <typename T>
class MSE : public BaseLossFunction<T> {
    public:
        T loss(const Matrix<T>& prediction, const Matrix<T>& target) override {
            if (prediction.rows != target.rows || prediction.columns != target.columns) {
                throw std::runtime_error("Cannot calculate MSE Loss due to matrix size mismatch.");
            }
            
            T total_elements = static_cast<T>(target.rows * target.columns);
            T loss = static_cast<T>(0);

            for (std::size_t i = 0; i < prediction.rows; ++i) {
                for (std::size_t j = 0; j < prediction.columns; ++j) {
                    T difference = prediction.get(i, j) - target.get(i, j);
                    loss += (difference * difference);
                }
            }

            return (loss / total_elements);
        }

        Matrix<T> gradient(const Matrix<T>& prediction, const Matrix<T>& target) override {
            if (prediction.rows != target.rows || prediction.columns != target.columns) {
                throw std::runtime_error("Cannot calculate MSE Gradient due to matrix size mismatch.");
            }

            Matrix<T> calculated_gradient(prediction.rows, target.columns);
            std::size_t total_elements = (prediction.rows * prediction.columns);

            for (std::size_t i = 0; i < prediction.rows; ++i) {
                for (std::size_t j = 0; j < prediction.columns; ++j) {
                    T predicted_value = prediction.get(i, j);
                    T target_value = target.get(i, j);

                    calculated_gradient.set(i, j, (static_cast<T>(2) * (predicted_value - target_value)) / total_elements);
                }
            }

            return calculated_gradient;
        }
};

template <typename T>
class CategoricalCrossEntropy : public BaseLossFunction<T> {
    public:
        T loss(const Matrix<T>& prediction, const Matrix<T>& target) override {
            if (prediction.rows != target.rows || prediction.columns != target.columns) {
                throw std::runtime_error("Cannot calculate CCE Loss due to matrix size mismatch.");
            }

            std::size_t N = prediction.rows;
            T loss = static_cast<T>(0);

            for (std::size_t i = 0; i < prediction.rows; ++i) {
                for (std::size_t j = 0; j < prediction.columns; ++j) {
                    const T epsilon = static_cast<T>(1e-12);
                    T p = prediction.get(i, j);

                    if (p < epsilon) {
                        p = epsilon;
                    }

                    if (p > static_cast<T>(1) - epsilon) {
                        p = static_cast<T>(1) - epsilon;
                    }

                    loss += target.get(i, j) * std::log(p);
                }
            }

            return (static_cast<T>(-1) / N) * loss;
        }

        Matrix<T> gradient(const Matrix<T>& prediction, const Matrix<T>& target) override {
            if (prediction.rows != target.rows || prediction.columns != target.columns) {
                throw std::runtime_error("Cannot calculate CCE Gradient due to matrix size mismatch.");
            }

            Matrix<T> calculated_gradient(prediction.rows, target.columns);
            std::size_t N = prediction.rows;

            for (std::size_t i = 0; i < prediction.rows; ++i) {
                for (std::size_t j = 0; j < prediction.columns; ++j) {
                    calculated_gradient.set(i, j, (prediction.get(i, j) - target.get(i, j)) / N);
                }
            }

            return calculated_gradient;
        }
};