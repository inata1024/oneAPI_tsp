#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<cmath>
#include <chrono>
#include <omp.h>
using namespace std;

void TSP(vector<vector<int> > W) {
    int n = W.size(); // 顶点数量

    // 初始化参数
    int startT = 3000; 
    double endT = 1e-8;
    double delta = 0.999; 
    int limit = 10000;   

    vector<int> path;
    int length_sum = 0;

    for (int i = 0; i < n; i++) {
        path.push_back(i);
    }

    // 计算初始最优路径和
    for (int i = 0; i < n - 1; i++) {
        length_sum += W[path[i]][path[i + 1]];
    }

    length_sum += W[path[n - 1]][path[0]];

    // 使用蒙特卡洛得到一个较好的初始解
    vector<int> cur = path;
    int cur_sum = 0;

    for (int i = 0; i < 8000; i++) {
        for (int k = 0; k < n; k++) {
            int j = rand() % n;
            swap(cur[k], cur[j]);
        }
        #pragma omp parallel for reduction(+:cur_sum)
        for (int i = 0; i < n - 1; i++) {
            cur_sum += W[cur[i]][cur[i + 1]];
        }

        cur_sum += W[cur[n - 1]][cur[0]];

        if (cur_sum < length_sum) {
            path = cur;
            length_sum = cur_sum;
        }
    }

    // 退火模拟过程
    srand((int)(time(NULL)));

    while (startT > endT) {
        vector<int> path_new = path;
        int length_sum_new = 0;
        int P_L = 0;

        int x = rand() % n;
        int y = rand() % n;

        while (x == y) {
            x = rand() % n;
            y = rand() % n;
        }

        swap(path_new[x], path_new[y]);

        //#pragma omp parallel for reduction(+:length_sum_new)
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
}

int main() {
    //手动调整n值以测试算法性能
    int n = 10000;
    vector<vector<int> > w(n, vector<int>(n));

    // 随机生成邻接矩阵
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                w[i][j] = rand() % 100 + 1; 
            } else {
                w[i][j] = 0;
            }
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    TSP(w);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    cout << fp_ms.count() << "ms" << endl;
    return 0;
}
