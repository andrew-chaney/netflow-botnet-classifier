import time


def function_timer(func):
    def decorator(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__}() executed in {(t2-t1):.6f}s")
        return result
    return decorator
