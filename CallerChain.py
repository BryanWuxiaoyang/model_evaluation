class CallerChain:
    """
    Helper class that can help linking actions.
    """
    def __init__(self, *callers):
        self.__callers = callers
        self.__next_caller = None

    def set_next(self, next_caller):
        self.__next_caller = next_caller

    def __call__(self, *args, **kwargs):
        for caller in self.__callers:
            caller(*args, **kwargs)
        if self.__next_caller is not None:
            self.__next_caller(*args, **kwargs)


if __name__ == "__main__":
    caller = CallerChain(1, 2)

