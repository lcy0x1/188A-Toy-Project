# this code is a modification of the heapq.py code

class ArrHeap:

    def __init__(self, n, ini):
        self.n = n
        self.array = [ini for _ in range(n)]  # array_index to value
        self.reverse = [i for i in range(n)]  # array_index to heap_index
        self.heap = [i for i in range(n)]  # heap_index to array_index

    def to_obj(self):
        return {
            'a': self.array,
            'r': self.reverse,
            'h': self.heap
        }

    def read(self, data):
        self.array = data['a']
        self.reverse = data['r']
        self.heap = data['h']

    def update(self, i, v):
        self.array[i] = v
        self._move_down(self.reverse[i])

    def get_max(self):
        return self.array[self.heap[0]]

    def get_index_of_max(self):
        return self.heap[0]

    def get(self, i):
        return self.array[i]

    def _move_up(self, startpos, pos):
        newitem = self.heap[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if self.array[parent] < self.array[newitem]:
                self._exchange(pos, parentpos)
                pos = parentpos
                continue
            break

    def _move_down(self, pos):
        endpos = self.n
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and not self.array[self.heap[rightpos]] < self.array[self.heap[childpos]]:
                childpos = rightpos
            self._exchange(pos, childpos)
            pos = childpos
            childpos = 2 * pos + 1
        self._move_up(0, pos)

    def _exchange(self, a, b):
        ai_a = self.heap[a]
        ai_b = self.heap[b]
        self.heap[a] = ai_b
        self.heap[b] = ai_a
        self.reverse[ai_a] = b
        self.reverse[ai_b] = a
