class SimpleExecutor:
    """
    A serial executor, applied to integrate into codes together with parallel executors such as [ThreadPoolExecutor].
    """
    class SimpleFuture:
        def __init__(self, ret):
            self.ret = ret

        def result(self):
            return self.ret

    def submit(self, runner, *args, **kwargs):
        return SimpleExecutor.SimpleFuture(runner(*args, **kwargs))