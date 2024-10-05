# merge_sort.py

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result

def merge_sort_in_place(arr):
    _merge_sort_in_place(arr, 0, len(arr) - 1)

def _merge_sort_in_place(arr, low, high):
    if low < high:
        mid = (low + high) // 2
        _merge_sort_in_place(arr, low, mid)
        _merge_sort_in_place(arr, mid + 1, high)
        _merge_in_place(arr, low, mid, high)

def _merge_in_place(arr, low, mid, high):
    left = arr[low:mid + 1]
    right = arr[mid + 1:high + 1]
    i = j = 0
    k = low
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

def merge_sort_iterative(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            mid = (low + high) // 2
            stack.append((low, mid))
            stack.append((mid + 1, high))
    return arr

def merge_sort_parallel(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort_parallel(arr[:mid])
    right = merge_sort_parallel(arr[mid:])
    return merge(left, right)

def merge_sort_hybrid(arr):
    if len(arr) <= 10:
        return sorted(arr)
    mid = len(arr) // 2
    left = merge_sort_hybrid(arr[:mid])
    right = merge_sort_hybrid(arr[mid:])
    return merge(left, right)

# Example usage:
arr = [5, 2, 9, 1, 7, 3, 6, 8, 4]
print(merge_sort(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(merge_sort_in_place(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(merge_sort_iterative(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(merge_sort_parallel(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(merge_sort_hybrid(arr))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
