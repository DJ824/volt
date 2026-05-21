#include <iostream>
#include "include/thread_pool.h"
// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.

int main() {
    ThreadPool pool{4};
    pool.start();

    auto f = pool.submit([](int a, int b) {
        return a + b;
    }, 4, 5);




    std::cout << f.get() << std::endl;

}