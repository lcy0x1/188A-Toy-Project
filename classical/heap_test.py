import random

from array_heap import ArrHeap

if __name__ == '__main__':
    n = 100
    heap = ArrHeap(n, 0)
    m = 10000
    ans = True
    for _ in range(m):
        ind = random.randrange(n)
        val = random.random() * 10 - 5
        heap.update(ind, val)
        ans = ans and (heap.get_max() == max(heap.array))
        if not ans:
            print(ans)
            print(heap.array)
            print(heap.heap)
            print(heap.reverse)
            break
    print(ans)
