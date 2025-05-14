from multiprocessing import Queue
from typing import Any, Callable, NoReturn

if not hasattr(Queue, '__getitem__'):
    class SubscriptableMethodProxy:
        __origin__: Callable[..., Any]
        __args__: tuple[type, ...] | None
        
        # Queue[T] can make an error in runtime, this wrapper can make it work
        def __init__(self, method: Callable[..., Any], _type: type | None = None):
            # method = Queue
            self.__origin__ = method
            self.__args__ = (_type,) if _type is not None else None

        def __call__(self, *args: Any, **kwargs: Any):
            # Call the original method to create a queue
            return self.__origin__(*args, **kwargs)
        
        def __repr__(self):
            if self.__args__ is None:
                return f"{self.__origin__.__name__}()"
            return f"{self.__origin__.__name__}[{self.__args__[0].__name__}]"

        def __getitem__(self, _type: type):
            if self.__args__ is not None:
                raise TypeError(f"Cannot subscript twice: {self.__origin__.__name__}[{self.__args__[0].__name__}]")
            return SubscriptableMethodProxy(self.__origin__, _type)
        
        def __or__(self, other: Any) -> Any:
            pass
        
        def __ror__(self, other: Any) -> Any:
            pass
    Queue = SubscriptableMethodProxy(Queue)
    # This function exists only to fool the type checker so that it believes
    # that `Queue` is not changed to something else than a class
    def suppress_queue_type_check_error() -> NoReturn: ...
    suppress_queue_type_check_error()
else:
    # Pretending import to fool the type checker
    from multiprocessing import Queue