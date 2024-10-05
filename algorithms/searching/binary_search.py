# binary_search.py

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def binary_search_recursive(arr, target):
    if len(arr) == 0:
        return -1
    mid = len(arr) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr[mid + 1:], target) + mid + 1
    else:
        return binary_search_recursive(arr[:mid], target)

def binary_search_iterative(arr, target):
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            stack.append((mid + 1, high))
        else:
            stack.append((low, mid - 1))
    return -1

def binary_search_parallel(arr, target):
    if len(arr) == 0:
        return -1
    mid = len(arr) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_parallel(arr[mid + 1:], target) + mid + 1
    else:
        return binary_search_parallel(arr[:mid], target)

def binary_search_hybrid(arr, target):
    if len(arr) <= 10:
        return linear_search(arr, target)
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Example usage:
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(binary_search(arr, 5))  # 4
print(binary_search_recursive(arr, 5))  # 4
print(binary_search_iterative(arr, 5))  # 4
print(binary_search_parallel(arr, 5))  # 4
print(binary_search_hybrid(arr, 5))  # 4
