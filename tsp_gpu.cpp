#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<cmath>
#include <sycl/sycl.hpp>
using namespace std;
static constexpr size_t n = 200;

int main() {

    sycl::queue queue;
    std::cout << "Device : " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    // 为共享变量分配空间
    auto cur = malloc_shared<int>(N + 1, q);
	for(int i = 0;i < n;i++)
		cur[i] = path[i];
	auto W = malloc_shared<int*>(N, q);
	for(int i=0;i<N;i++)
		W[i] = malloc_shared<int>(N, q);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i != j) {
				W[i][j] = rand() % 100 + 1;
			} else {
				W[i][j] = 0;
			}
		}
	}
	
	auto t1 = std::chrono::high_resolution_clock::now();

    // 初始化参数
    int startT = 3000; 
    double endT = 1e-8; 
    double delta = 0.999;
    int limit = 10000;    

	int* path = new int[n];
    int length_sum = 0;

    for (int i = 0; i < n; i++) {
        path[i] = i;
    }

    // 计算初始最优路径和
    for (int i = 0; i < n - 1; i++) {
        length_sum += W[path[i]][path[i + 1]];
    }

    length_sum += W[path[n - 1]][path[0]];

    int cur_sum = 0;

    for (int i = 0; i < 8000; i++) {
        for (int k = 0; k < n; k++) {
            int j = rand() % n;
            swap(cur[k], cur[j]);
        }

        //--------------------
        // 规约操作卸载到设备端  
        queue.single_task([=]() {
            int curr = 0;
            for (int i = 0; i < n - 1; i++) {
                curr += W[cur[i]][cur[i + 1]];
            }
            cur[n] = curr;
        }).wait();
        //--------------------
        cur_sum = cur[n];
        cur_sum += W[cur[n - 1]][cur[0]];

        if (cur_sum < length_sum) {
            path = cur;
            length_sum = cur_sum;
        }
    }
    

    // 退火模拟过程
    srand((int)(time(NULL)));

    while (startT > endT) {
        int* path_new = new int[n];
		for(int i = 0;i < n;i++)
			path_new[i] = path[i];
        int length_sum_new = 0;
        int P_L = 0;

        int x = rand() % n;
        int y = rand() % n;

        while (x == y) {
            x = rand() % n;
            y = rand() % n;
        }

        swap(path_new[x], path_new[y]);
        for (int i = 0; i < n - 1; i++) {
            length_sum_new += W[path_new[i]][path_new[i + 1]];
        }

        length_sum_new += W[path_new[n - 1]][path_new[0]];

        double dE = length_sum_new - length_sum;

        if (dE < 0) {
            path = path_new;
            length_sum = length_sum_new;
        } else {
            double rd = rand() / (RAND_MAX + 1.0);
            if (exp(-dE / startT) > rd) {
                path = path_new;
                length_sum = length_sum_new;
                P_L++;
            }
        }

        if (P_L == limit) break;

        startT *= delta;
    }

	// 释放空间
    sycl::free(cur, queue);
    sycl::free(data, queue);

	auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    cout << fp_ms.count() << "ms" << endl;
    return 0;
}