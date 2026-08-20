#pragma once

#include <memory>

#include "layers.hpp"
#include "loss.hpp"

template <typename T>
struct TrainStepResult {
    double accuracy;
    T loss;
};

template <typename T>
class Network {
    private:
        std::vector<std::unique_ptr<BaseLayer<T>>> layers;
        std::unique_ptr<BaseLossFunction<T>> lossFunction;

        std::size_t output_size;
        std::size_t input_size;

    public:
        Network(std::size_t input_size, std::size_t output_size, std::unique_ptr<BaseLossFunction<T>> lossFunction) :
            output_size(output_size),
            input_size(input_size),
            lossFunction(std::move(lossFunction)) {}

        void add_layer(std::unique_ptr<BaseLayer<T>> layer) {
            layers.push_back(std::move(layer));
        }

        Matrix<T> forward(const Matrix<T>& input) {
            Matrix<T> x = input;

            for (auto& layer : layers) {
                x = layer->forward(x);
            }

            return x;
        }

        TrainStepResult<T> train_step(const Matrix<T>& input, const Matrix<T>& expected, double learning_rate) {
            TrainStepResult<T> result;

            Matrix<T> predicted = forward(input);
            Matrix<T> loss_gradient = lossFunction->gradient(predicted, expected);
            T loss = lossFunction->loss(predicted, expected);
            double acc = accuracy(predicted, expected);

            for (int i = layers.size() - 1; i >= 0; --i) {
                loss_gradient = layers[i]->backward(loss_gradient);
            }

            for (auto& layer : layers) {
                layer->update(learning_rate);
            }

            result.accuracy = acc;
            result.loss = loss;

            return result;
        }

        T accuracy(const Matrix<T>& prediction, const Matrix<T>& target) {
            std::size_t correct = 0;

            for (std::size_t i = 0; i < prediction.rows; ++i) {
                std::size_t predicted_class = 0;
                std::size_t target_class = 0;

                for (std::size_t j = 1; j < prediction.columns; ++j) {
                    if (prediction.get(i, j) > prediction.get(i, predicted_class)) {
                        predicted_class = j;
                    }

                    if (target.get(i, j) > target.get(i, target_class)) {
                        target_class = j;
                    }
                }

                if (predicted_class == target_class) {
                    ++correct;
                }
            }

            return static_cast<double>(correct) / static_cast<double>(prediction.rows);
        }
};