# NPU-System-Simulator
NPU-System-Simulator : 
NPU Performance & Karatsuba Multiplication
This repository showcases a simplified Neural Processing Unit (NPU) environment in C++ to explore different multiplication methods (Classical vs. Karatsuba) and vectorized operations (like SoftMax) for deep learning workloads. It also includes a Python script for plotting performance metrics from logged CSV data.

Features
Karatsuba vs. Classical Multiplication

Demonstrates fast integer multiplication with Karatsuba, highlighting its performance gains over the traditional approach.

Evaluates execution time, throughput, and clock cycles under varying workloads.

Vector Processing Unit (VPU)

Implements vectorized operations including an optimized SoftMax function.

Measures VPU overhead and shows how different vector lengths (256, 512, 1024, 2048) impact performance.

Task Scheduling & Multi-Threading

A simple queue-based scheduler assigns workloads (Karatsuba, Classical, or VPU tasks).

Multiple threads process tasks in parallel, tracking overall execution times.

Performance Logging

Results (execution time, throughput, cycles) are appended to npu_statistics.csv.

Allows you to compare multiple runs or configurations over time.

Data Visualization

plots.py reads from npu_statistics.csv and generates performance charts to visualize workload vs. execution time.
Notes & Best Practices
The Karatsuba algorithm provides an efficiency boost especially for large inputs. Smaller numbers won’t show as dramatic a difference.

Larger vector lengths (e.g., 2048) reduce overall tasks at the cost of more hardware resources. The 1024 length often gives a balanced trade-off.

Adjust workload sizes or vector lengths in NPU.cpp to replicate different scenarios.

Contributing
Feel free to open issues or pull requests! This code can be extended to explore:

Mixed-precision arithmetic (FP16/INT8).

Additional activation or matrix-ops in the VPU.

Larger simulation frameworks (e.g., SystemC).
