# linear_search.py

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def linear_search_recursive(arr, target):
    if len(arr) == 0:
        return -1
    if arr[0] == target:
        return 0
    return linear_search_recursive(arr[1:], target) + 1

def linear_search_iterative(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def linear_search_parallel(arr, target):
    if len(arr) == 0:
        return -1
    if arr[0] == target:
        return 0
    return linear_search_parallel(arr[1:], target) + 1

def linear_search_hybrid(arr, target):
    if len(arr) <= 10:
        return linear_search(arr, target)
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def linear_search_with_sentinel(arr, target):
    arr.append(target)
    i = 0
    while arr[i] != target:
        i += 1
    if i == len(arr) - 1:
        return -1
    return i

def linear_search_with_hashing(arr, target):
    hash_table = {}
    for i in range(len(arr)):
        hash_table[arr[i]] = i
    return hash_table.get(target, -1)

# Example usage:
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(linear_search(arr, 5))  # 4
print(linear_search_recursive(arr, 5))  # 4
print(linear_search_iterative(arr, 5))  # 4
print(linear_search_parallel(arr, 5))  # 4
print(linear_search_hybrid(arr, 5))  # 4
print(linear_search_with_sentinel(arr, 5))  # 4
print(linear_search_with_hashing(arr, 5))  # 4
