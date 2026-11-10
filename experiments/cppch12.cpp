#include <iostream>
#include <vector>



int main() {
    std::vector<int> nums = {10, 20, 30};

    nums.push_back(40);
    nums.pop_back();

    for (int x : nums) {
        std::cout << x << " ";
    }
}