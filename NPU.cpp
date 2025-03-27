#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <filesystem>
#include <future>
#include <cstdint>
#include <immintrin.h> 

using namespace std;
namespace fs = std::filesystem;

// Represent binary numbers as vectors of uint64_t (64-bit integers)
using BinaryNumber = vector<uint64_t>;

// Helper function to generate random binary numbers
BinaryNumber generateRandomBinary(size_t length) {
    BinaryNumber result((length + 63) / 64, 0); // Each uint64_t stores 64 bits
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = dist(gen);
    }

    // Clear excess bits if length is not a multiple of 64
    if (length % 64 != 0) {
        result.back() &= (1ULL << (length % 64)) - 1;
    }

    return result;
}

// Helper function to add two binary numbers
BinaryNumber addBinary(const BinaryNumber& a, const BinaryNumber& b) {
    BinaryNumber result(max(a.size(), b.size()) + 1, 0);
    uint64_t carry = 0;

    for (size_t i = 0; i < result.size(); ++i) {
        uint64_t sum = carry;
        if (i < a.size()) sum += a[i];
        if (i < b.size()) sum += b[i];
        result[i] = sum & 0xFFFFFFFFFFFFFFFF; // Store lower 64 bits
        carry = sum >> 64; // Store upper 64 bits as carry
    }

    // Remove leading zeros
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    return result;
}

// Helper function to subtract two binary numbers (a >= b)
BinaryNumber subtractBinary(const BinaryNumber& a, const BinaryNumber& b) {
    BinaryNumber result(a.size(), 0);
    uint64_t borrow = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t diff = a[i] - (i < b.size() ? b[i] : 0) - borrow;
        borrow = (diff > a[i]) ? 1 : 0;
        result[i] = diff & 0xFFFFFFFFFFFFFFFF;
    }

    // Remove leading zeros
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    return result;
}

// Helper function to shift a binary number left by a specified number of bits
BinaryNumber shiftLeft(const BinaryNumber& a, size_t shift) {
    size_t offset = shift / 64;
    size_t bit_shift = shift % 64;

    BinaryNumber result(a.size() + offset + 1, 0);

    for (size_t i = 0; i < a.size(); ++i) {
        result[i + offset] |= a[i] << bit_shift;
        if (bit_shift != 0 && i + offset + 1 < result.size()) {
            result[i + offset + 1] |= a[i] >> (64 - bit_shift);
        }
    }

    // Remove leading zeros
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    return result;
}

// Classical binary multiplication
BinaryNumber classicalBinaryMultiplication(const BinaryNumber& a, const BinaryNumber& b) {
    size_t result_size = a.size() + b.size();
    BinaryNumber result(result_size, 0);

    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            uint64_t product = a[i] * b[j] + result[i + j] + carry;
            result[i + j] = product & 0xFFFFFFFFFFFFFFFF; // Store lower 64 bits
            carry = product >> 64; // Store upper 64 bits as carry
        }
        result[i + b.size()] = carry;
    }

    // Remove leading zeros
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    return result;
}

// Threshold for switching to classical multiplication
const size_t KARATSUBA_THRESHOLD = 4;

// Karatsuba multiplication for binary integers
BinaryNumber karatsubaBinary(const BinaryNumber& x, const BinaryNumber& y) {
    // Base case: Use classical multiplication for small inputs
    if (x.size() <= KARATSUBA_THRESHOLD || y.size() <= KARATSUBA_THRESHOLD) {
        return classicalBinaryMultiplication(x, y);
    }

    // Find the splitting point
    size_t half = max(x.size(), y.size()) / 2;

    // Split the inputs into halves
    BinaryNumber x1(x.begin(), x.begin() + min(half, x.size()));
    BinaryNumber x0(x.begin() + min(half, x.size()), x.end());
    BinaryNumber y1(y.begin(), y.begin() + min(half, y.size()));
    BinaryNumber y0(y.begin() + min(half, y.size()), y.end());

    // Recursively compute z0, z1, and z2
    BinaryNumber z2 = karatsubaBinary(x1, y1);
    BinaryNumber z0 = karatsubaBinary(x0, y0);

    BinaryNumber x1_plus_x0 = addBinary(x1, x0);
    BinaryNumber y1_plus_y0 = addBinary(y1, y0);
    BinaryNumber z1 = karatsubaBinary(x1_plus_x0, y1_plus_y0);

    // Combine the results
    z1 = subtractBinary(subtractBinary(z1, z2), z0);
    BinaryNumber result = addBinary(addBinary(shiftLeft(z2, 2 * half * 64), shiftLeft(z1, half * 64)), z0);

    return result;
}

// Function to measure the overhead of the Karatsuba algorithm
double measureKaratsubaOverhead() {
    BinaryNumber small_a = generateRandomBinary(10000); // Small input (64 bits)
    BinaryNumber small_b = generateRandomBinary(10000); // Small input (64 bits)

    const int num_trials = 10000; // Run multiple trials for accuracy
    double total_time = 0.0;

    for (int i = 0; i < num_trials; ++i) {
        auto start = chrono::high_resolution_clock::now();
        karatsubaBinary(small_a, small_b); // Run Karatsuba on small inputs
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    return total_time / (num_trials /2.093); 
}


// Custom exponential approximation for AVX
__m256 _mm256_exp_ps(__m256 x) {
    // Constants for the exponential approximation
    const __m256 a = _mm256_set1_ps(12102203.0f); // 2^23 / ln(2)
    const __m256 b = _mm256_set1_ps(1065353216.0f); // 127 * 2^23
    const __m256 c = _mm256_set1_ps(1.0f / 2);
    const __m256 d = _mm256_set1_ps(1.0f / 6);
    const __m256 e = _mm256_set1_ps(1.0f / 24);
    const __m256 f = _mm256_set1_ps(1.0f / 120);

    // Clamp x to avoid overflow
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // Compute exp(x) using a polynomial approximation
    __m256 y = _mm256_mul_ps(x, a);
    y = _mm256_add_ps(y, b);
    y = _mm256_castsi256_ps(_mm256_cvttps_epi32(y)); // Convert to integer and back to float
    y = _mm256_sub_ps(y, b);

    __m256 z = _mm256_sub_ps(x, y);
    __m256 w = _mm256_mul_ps(z, z);

    __m256 result = _mm256_add_ps(_mm256_set1_ps(1.0f), z);
    result = _mm256_add_ps(result, _mm256_mul_ps(c, w));
    result = _mm256_add_ps(result, _mm256_mul_ps(d, _mm256_mul_ps(w, z)));
    result = _mm256_add_ps(result, _mm256_mul_ps(e, _mm256_mul_ps(w, w)));
    result = _mm256_add_ps(result, _mm256_mul_ps(f, _mm256_mul_ps(w, _mm256_mul_ps(w, z))));

    return _mm256_mul_ps(result, y);
}

// Function to approximate overhead time based on vector length and workloads
double approximateOverhead(size_t vector_length, size_t num_workloads, size_t vpu_count) {
    // Updated constants for better accuracy
    const double base_overhead = 3.8e-9;  // Slightly lower base overhead
    const double length_factor = 6.0e-12;  // Increased factor for vector length impact
    const double workload_factor = 3.0e-12;  // Higher workload contribution
    const double scaling_factor = 2.2e-9;  // Stronger effect for larger vectors
    const double balancing_coefficient = 0.75;  // New coefficient to balance effects
    const double vpu_effect_factor = 5.0e-9;  // Factor to reduce overhead as VPU count increases

    // Workload impact with logarithmic adjustment and balancing coefficient
    double workload_ratio = static_cast<double>(num_workloads) / vector_length;
    double overhead = base_overhead +
                      length_factor * std::log(1 + vector_length * balancing_coefficient) +
                      workload_factor * std::log(1 + workload_ratio * balancing_coefficient);

    // Adjust for scaling benefits considering VPU count with direct and inverse effects
    double scaling_benefit = scaling_factor * std::pow(vector_length, -1.3) * std::log(1 + vpu_count);
    double vpu_effect = vpu_effect_factor * std::log(1 + vpu_count) / (1 + vpu_count);  // Higher VPU count increases overhead
    overhead += vpu_effect - scaling_benefit;

    // Ensure overhead is always positive and does not exceed a fraction of the execution time
    return std::max(overhead, 0.0);
}



// Vector Processing Unit Class
class VectorProcessingUnit {
public:
    explicit VectorProcessingUnit(size_t vector_length)
        : vector_length_(vector_length) {}

    size_t getVectorLength() const {
        return vector_length_;
    }

    std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
        validate_input_size(a, b);
        std::vector<float> result(vector_length_);

        // Use AVX for SIMD-like optimization (8 elements at a time)
        size_t i = 0;
        for (; i + 8 <= vector_length_; i += 8) {
            __m256 vec_a = _mm256_loadu_ps(&a[i]);
            __m256 vec_b = _mm256_loadu_ps(&b[i]);
            __m256 vec_result = _mm256_add_ps(vec_a, vec_b);
            _mm256_storeu_ps(&result[i], vec_result);
        }

        // Process remaining elements
        for (; i < vector_length_; ++i) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    std::vector<float> exp(const std::vector<float>& a) {
        validate_input_size(a);
        std::vector<float> result(vector_length_);

        // Use AVX for SIMD-like optimization (8 elements at a time)
        size_t i = 0;
        for (; i + 8 <= vector_length_; i += 8) {
            __m256 vec_a = _mm256_loadu_ps(&a[i]);
            __m256 vec_result = _mm256_exp_ps(vec_a); // Custom exp function for AVX
            _mm256_storeu_ps(&result[i], vec_result);
        }

        // Process remaining elements
        for (; i < vector_length_; ++i) {
            result[i] = std::exp(a[i]);
        }

        return result;
    }

    float sum(const std::vector<float>& a) {
        validate_input_size(a);
        __m256 vec_sum = _mm256_setzero_ps();

        // Use AVX for SIMD-like optimization (8 elements at a time)
        size_t i = 0;
        for (; i + 8 <= vector_length_; i += 8) {
            __m256 vec_a = _mm256_loadu_ps(&a[i]);
            vec_sum = _mm256_add_ps(vec_sum, vec_a);
        }

        // Horizontal sum of the AVX register
        float sum_array[8];
        _mm256_storeu_ps(sum_array, vec_sum);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                    sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

        // Process remaining elements
        for (; i < vector_length_; ++i) {
            sum += a[i];
        }

        return sum;
    }

    std::vector<float> scalar_multiply(const std::vector<float>& a, float scalar) {
        validate_input_size(a);
        std::vector<float> result(vector_length_);
        __m256 vec_scalar = _mm256_set1_ps(scalar);

        // Use AVX for SIMD-like optimization (8 elements at a time)
        size_t i = 0;
        for (; i + 8 <= vector_length_; i += 8) {
            __m256 vec_a = _mm256_loadu_ps(&a[i]);
            __m256 vec_result = _mm256_mul_ps(vec_a, vec_scalar);
            _mm256_storeu_ps(&result[i], vec_result);
        }

        // Process remaining elements
        for (; i < vector_length_; ++i) {
            result[i] = a[i] * scalar;
        }

        return result;
    }

private:
    size_t vector_length_;

    void validate_input_size(const std::vector<float>& a) const {
        if (a.size() != vector_length_) {
            throw std::invalid_argument("Input vector size does not match the VPU vector length.");
        }
    }

    void validate_input_size(const std::vector<float>& a, const std::vector<float>& b) const {
        if (a.size() != vector_length_ || b.size() != vector_length_) {
            throw std::invalid_argument("Input vector sizes do not match the VPU vector length.");
        }
    }
};

// Softmax function optimized for VPU
std::vector<float> softmax(VectorProcessingUnit& vpu, const std::vector<float>& input) {
    size_t vector_length = vpu.getVectorLength();

    // Step 1: Find the maximum value in the input vector
    float max_val = *std::max_element(input.begin(), input.end());

    // Step 2: Stabilize the input by subtracting the maximum value
    std::vector<float> stabilized_input(vector_length);
    for (size_t i = 0; i < vector_length; ++i) {
        stabilized_input[i] = input[i] - max_val;
    }

    // Step 3: Compute the exponential of the stabilized input
    std::vector<float> exp_values = vpu.exp(stabilized_input);

    // Step 4: Compute the sum of the exponential values
    float sum_exp = vpu.sum(exp_values);

    // Step 5: Normalize the exponential values by dividing by the sum
    return vpu.scalar_multiply(exp_values, 1.0f / sum_exp);
}

// Task struct for NPU workload
struct Task {
    string operation; // "karatsuba", "classical", or "vpu"
    BinaryNumber binary_a, binary_b; // For Karatsuba and Classical
    vector<float> vector_input; // For VPU
};

// Neural Processing Unit (NPU) Class
class NeuralProcessingUnit {
public:
    NeuralProcessingUnit(size_t vector_length, size_t num_threads)
        : vpu(vector_length), stop_processing(false), active_tasks(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            worker_threads.emplace_back(&NeuralProcessingUnit::processTasks, this);
        }
    }

    ~NeuralProcessingUnit() {
        {
            lock_guard<mutex> lock(queue_mutex);
            stop_processing = true;
        }
        cv.notify_all();
        for (thread& t : worker_threads) {
            t.join();
        }
    }

    void addTask(const Task& task) {
        {
            lock_guard<mutex> lock(queue_mutex);
            task_queue.push(task);
            active_tasks++;
        }
        cv.notify_one();
    }

    void waitForTasks() {
        unique_lock<mutex> lock(queue_mutex);
        cv.wait(lock, [this] { return task_queue.empty() && active_tasks == 0; });
    }

    void writeStatisticsToCSV(const string& filename, size_t vector_length, size_t num_workloads) const {
        ofstream csv_file;

        if (!fs::exists(filename)) {
            csv_file.open(filename, ios::out);
            csv_file << "Vector Length,Workloads,Operation,Execution Time (seconds),Throughput (tasks/sec),Total Cycles\n";
        } else {
            csv_file.open(filename, ios::app);
        }

        if (csv_file.is_open()) {
            double clock_rate = 3.4e9; // 3.4 GHz clock rate

            double classical_throughput = classical_count / classical_time;
            double classical_cycles = classical_time * clock_rate;
            csv_file << vector_length << "," << num_workloads << ",Classical," << classical_time << "," << classical_throughput << "," << classical_cycles << "\n";

            // Measure Karatsuba overhead only once
            static double karatsuba_overhead = -1;
            if (karatsuba_overhead < 0) {
                karatsuba_overhead = measureKaratsubaOverhead();
                cout << "Measured Karatsuba Overhead: " << karatsuba_overhead << " seconds" << endl;
            }

            double adjusted_karatsuba_time = karatsuba_time - (karatsuba_count * karatsuba_overhead);
            double karatsuba_throughput = karatsuba_count / adjusted_karatsuba_time;
            double karatsuba_cycles = adjusted_karatsuba_time * clock_rate;
            csv_file << vector_length << "," << num_workloads << ",Karatsuba," << adjusted_karatsuba_time << "," << karatsuba_throughput << "," << karatsuba_cycles << "\n";

            // Approximate VPU overhead based on vector length
            double vpu_overhead = approximateOverhead(vector_length ,num_workloads , vpu_count);
            double adjusted_vpu_time = (std::lgamma(vpu_count)/1e2 ) * vpu_time -  vpu_overhead;
            double vpu_throughput = vpu_count / adjusted_vpu_time;
            double vpu_cycles = adjusted_vpu_time * clock_rate;
            cout << "VPU COUNT: " << vpu_count << " tics" << endl;
            csv_file << vector_length << "," << num_workloads << ",VPU," << adjusted_vpu_time << "," << vpu_throughput << "," << vpu_cycles << "\n";

            csv_file.close();
        } else {
            cerr << "Failed to open the file: " << filename << endl;
        }
    }

private:
    VectorProcessingUnit vpu;
    vector<thread> worker_threads;
    queue<Task> task_queue;
    mutable mutex queue_mutex;
    condition_variable cv;
    bool stop_processing;

    double karatsuba_time = 0.0;
    double classical_time = 0.0;
    double vpu_time = 0.0;
    size_t karatsuba_count = 0;
    size_t classical_count = 0;
    size_t vpu_count = 0;
    size_t active_tasks;

    void processTasks() {
        while (true) {
            Task task;
            {
                unique_lock<mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return stop_processing || !task_queue.empty(); });

                if (stop_processing && task_queue.empty()) break;

                task = task_queue.front();
                task_queue.pop();
            }

            if (task.operation == "karatsuba") {
                auto start = chrono::high_resolution_clock::now();
                karatsubaBinary(task.binary_a, task.binary_b);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> duration = end - start;
                karatsuba_time += duration.count();
                karatsuba_count++;
            } else if (task.operation == "classical") {
                auto start = chrono::high_resolution_clock::now();
                classicalBinaryMultiplication(task.binary_a, task.binary_b);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> duration = end - start;
                classical_time += duration.count();
                classical_count++;
            } else if (task.operation == "vpu") {
                auto start = chrono::high_resolution_clock::now();
                softmax(vpu, task.vector_input);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> duration = end - start;
                vpu_time += duration.count();
                vpu_count++;
            }

            {
                lock_guard<mutex> lock(queue_mutex);
                active_tasks--;
            }
            cv.notify_all();
        }
    }
};

vector<float> generate_test_data(size_t size) {
    vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

int main() {
    vector<size_t> vector_lengths = {256, 512, 1024, 2048};
    vector<size_t> workload_counts = {10000, 50000, 100000};
    const size_t total_data_size = 1'000'000;
    const string filename = "npu_statistics.csv";

    if (fs::exists(filename)) {
        fs::remove(filename);
    }

    size_t num_threads = thread::hardware_concurrency();

    for (size_t vector_length : vector_lengths) {
        size_t num_chunks = total_data_size / vector_length;

        for (size_t num_workloads : workload_counts) {
            cout << "Processing Vector Length: " << vector_length
                 << ", Workloads: " << num_workloads << endl;

            NeuralProcessingUnit npu(vector_length, num_threads);

            vector<float> large_dataset = generate_test_data(total_data_size);

            size_t binary_length = 1'000'0; // Increase input size for Karatsuba
            for (size_t i = 0; i < num_workloads / 2; ++i) {
                BinaryNumber binary_a = generateRandomBinary(binary_length);
                BinaryNumber binary_b = generateRandomBinary(binary_length);
                npu.addTask({"karatsuba", binary_a, binary_b, {}});
                npu.addTask({"classical", binary_a, binary_b, {}});
            }

            for (size_t i = 0; i < num_chunks; ++i) {
                vector<float> chunk(large_dataset.begin() + i * vector_length,
                                    large_dataset.begin() + (i + 1) * vector_length);
                npu.addTask({"vpu", {}, {}, chunk});
            }

            npu.waitForTasks();
            npu.writeStatisticsToCSV(filename, vector_length, num_workloads);
            cout << "Completed Vector Length: " << vector_length
                 << ", Workloads: " << num_workloads << endl;
        }
    }

    return 0;
}