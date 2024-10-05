# quick_sort.py

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_in_place(arr):
    _quick_sort_in_place(arr, 0, len(arr) - 1)

def _quick_sort_in_place(arr, low, high):
    if low < high:
        pivot_index = _partition(arr, low, high)
        _quick_sort_in_place(arr, low, pivot_index - 1)
        _quick_sort_in_place(arr, pivot_index + 1, high)

def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_iterative(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_index = _partition(arr, low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))
    return arr

def quick_sort_parallel(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_parallel(left) + middle + quick_sort_parallel(right)

def quick_sort_hybrid(arr):
    if len(arr) <= 10:
        return sorted(arr)
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_hybrid(left) + middle + quick_sort_hybrid(right)

# Example usage:
arr = [5, 2, 9, 1, 7, 3, 6, 8, 4]
print(quick_sort(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(quick_sort_in_place(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(quick_sort_iterative(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(quick_sort_parallel(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(quick_sort_hybrid(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
