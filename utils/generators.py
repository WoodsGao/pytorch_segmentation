from queue import Queue
from threading import Thread


class CachedGenerator:
    def __init__(self, generator, queue_len=0):
        self.generator = generator
        self.queue = Queue(queue_len)
        t = Thread(target=self.worker)
        t.setDaemon(True)
        t.start()

    def worker(self):
        for item in self.generator:
            self.queue.put(item)

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()

    def __len__(self):
        return len(self.generator)
