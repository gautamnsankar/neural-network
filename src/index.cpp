#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

#include "network.hpp"

struct MNISTSample {
    Matrix<double> input;
    Matrix<double> label;
};

std::vector<MNISTSample> load_mnist_csv(const std::string& path, int limit = 1000) {
    std::ifstream file(path);
    std::vector<MNISTSample> data;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty: " + path);
    }

    int count = 0;

    while (count < limit && std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> fields;

        while (std::getline(ss, value, ',')) {
            fields.push_back(value);
        }

        if (fields.size() != 785) {
            std::cout << "Skipping bad row with " << fields.size() << " fields. Expected 785.\n";
            continue;
        }

        int digit = std::stoi(fields[0]);

        Matrix<double> input(1, 784);
        Matrix<double> expected(1, 10);

        for (int j = 0; j < 10; ++j) {
            expected.set(0, j, 0.0);
        }
        expected.set(0, digit, 1.0);

        for (int j = 0; j < 784; ++j) {
            double pixel = std::stod(fields[j + 1]) / 255.0;
            input.set(0, j, pixel);
        }

        data.push_back({input, expected});
        count++;
    }

    std::cout << "Loaded " << data.size() << " samples\n";
    return data;
}

int main() {
    auto dataset = load_mnist_csv("mnist_train.csv", 1000);

    Network<double> n(784, 10, std::make_unique<CategoricalCrossEntropy<double>>());

    n.add_layer(std::make_unique<LinearLayer<double>>(784, 128));
    n.add_layer(std::make_unique<ReLU<double>>());

    n.add_layer(std::make_unique<LinearLayer<double>>(128, 64));
    n.add_layer(std::make_unique<ReLU<double>>());

    n.add_layer(std::make_unique<LinearLayer<double>>(64, 10));
    n.add_layer(std::make_unique<Softmax<double>>());

    const double learning_rate = 0.0005;
    const int epochs = 35;

    for (int e = 0; e < epochs; ++e) {
        double total_loss = 0.0;
        double total_accuracy = 0.0;

        for (std::size_t i = 0; i < dataset.size(); ++i) {
            auto result = n.train_step(dataset[i].input, dataset[i].label, learning_rate);
            total_loss += result.loss;
            total_accuracy += result.accuracy;
        }

        std::cout << "Epoch " << e << " | avg loss = " << (total_loss / dataset.size()) << " | avg accuracy = " << (total_accuracy / dataset.size()) << '\n';
    }

    n.forward(dataset[10].input).print();
    dataset[10].label.print();

    return 0;
}