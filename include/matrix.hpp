#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

template <typename T>
class Matrix {
    public:
        std::size_t columns;
        std::size_t rows;

        Matrix(std::size_t rows, std::size_t columns) : columns(columns), rows(rows), data_(columns * rows) {}
        Matrix(std::size_t rows, std::size_t columns, T initalizer) : columns(columns), rows(rows), data_(columns * rows, initalizer) {}
        Matrix() = default;

        static Matrix ones(std::size_t rows, std::size_t columns) {
            return Matrix<T>(rows, columns, 1);
        }

        static Matrix zeros(std::size_t rows, std::size_t columns) {
            return Matrix<T>(rows, columns, 0);
        }

        static Matrix random(std::size_t rows, std::size_t columns, T min = static_cast<T>(-1), T max = static_cast<T>(1)) {
            Matrix<T> result(rows, columns);

            static std::random_device rd;
            static std::mt19937 gen(rd());

            std::uniform_real_distribution<T> dist(min, max);

            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(i, j, dist(gen));
                }
            }

            return result;
        }

        Matrix operator+(const Matrix& other) {
            if (columns != other.columns || rows != other.rows) {
                throw std::runtime_error("Cannot add matrices due to size mismatch.");
            }

            Matrix<T> result(rows, columns);
            
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(i, j, get(i, j) + other.get(i, j));
                }
            }

            return result;
        }

        Matrix operator-(const Matrix& other) {
            if (columns != other.columns || rows != other.rows) {
                throw std::runtime_error("Cannot add matrices due to size mismatch.");
            }

            Matrix<T> result(rows, columns);
            
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(i, j, get(i, j) - other.get(i, j));
                }
            }

            return result;
        }

        Matrix operator*(T scalar) {
            Matrix<T> result(rows, columns);

             for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(i, j, get(i, j) * scalar);
                }
            }

            return result;
        }

        static Matrix multiply(const Matrix& a, const Matrix& b) {
            if (a.columns != b.rows) {
                throw std::runtime_error("Cannot multiply matrices due to shape mismatch.");
            }

            Matrix<T> result(a.rows, b.columns);

            for (std::size_t i = 0; i < a.rows; ++i) {
                for (std::size_t k  = 0; k < a.columns; ++k) {
                    T a_ik = a.get(i, k);

                    for (std::size_t j = 0; j < b.columns; ++j) {
                        result.set(i, j, result.get(i, j) + a_ik * b.get(k, j));
                    }
                }
            }

            return result;
        }

        Matrix<T> transpose() const {
            Matrix<T> result(columns, rows);

            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(j, i, get(i, j));
                }
            }

            return result;
        }

        template <typename Function>
        Matrix<T> apply(Function f) const {
            Matrix<T> result(rows, columns);

            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result.set(i, j, f((*this).get(i, j)));
                }
            }

            return result;
        }

        T get(std::size_t row, std::size_t column) const {
            return data_[get_index_(row, column)];
        }

        void set(std::size_t row, std::size_t column, T value) {
            std::size_t index = get_index_(row, column);
            data_[index] = value;
        }

        void print() const {
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    std::cout << std::setw(3) << std::left << get(i, j) << "| ";
                }

                std::cout << '\n';
            }

            std::cout << '\n';
        }

    private:
        std::size_t get_index_(std::size_t row, std::size_t column) const {
            return column + (columns * row);
        }

        std::vector<T> data_;
};
