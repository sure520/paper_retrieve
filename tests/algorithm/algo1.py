import math
# def mian(l, r):
#     ans = 0
#     for i in range(l, r+1):
#         sqrt_num = math.isqrt(i)
#         if sqrt_num * sqrt_num == i:
#             ans += 1
#     return ans

def get_feibo_num_index(k, i):
    if i <= k:
        return 1
    else:
        return sum([get_feibo_num_index(k, i-j) for j in range(1, k+1)])

def main(k, q):
    N = 10**9 +7
    query_nums = []
    for i in range(q):
        query_num = int(input())
        query_nums.append(query_num)
    for num in query_nums:
        print(get_feibo_num_index(k, num)%N)

if __name__ == "__main__":
    main(3, 5)

