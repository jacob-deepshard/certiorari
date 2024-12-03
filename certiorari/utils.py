import functools
import inspect
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, Optional, Self

from pydantic import UUID4, BaseModel, ConfigDict, Field
from typing_extensions import Unpack


class BaseEntity(BaseModel):
    store: ClassVar[dict[UUID4, Self]]

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    created_at: datetime = Field(default_factory=datetime.now)

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        super().__init_subclass__(**kwargs)
        cls.store = {}

    def __init__(self, **kwargs: Unpack[ConfigDict]):
        super().__init__(**kwargs)
        self.store[self.id] = self

    def __del__(self):
        del self.store[self.id]

    @classmethod
    def __class_getitem__(cls, id: UUID4 | str) -> Self:
        if isinstance(id, str):
            id = UUID4(id)
        return cls.store[id]


import truffle

class Tool(BaseModel):
    name: str
    description: str
    start: Callable
    stop: Optional[Callable] = None
    
    def __call__(self, *args, **kwargs):
        return self.start(*args, **kwargs)
    
    def on_stop(self, fn: Callable):
        self.stop = fn
        
    def register_with_truffle(self):
        if self.start and self.stop:
            truffle.register_tool(self.start, name=f'start_{self.name}')
            truffle.register_tool(self.stop, name=f'stop_{self.name}')
        elif self.start:
            truffle.register_tool(self.start, name=self.name)
        else:
            raise ValueError(f'Tool {self.name} must have a start method')

class App(BaseModel):

    name: str
    fullname: str
    description: str
    goal: str

    database: list[BaseEntity] = Field(default_factory=list)
    tools: list[Callable] = Field(default_factory=list)

    @property
    def metadata(self):
        return truffle.AppMetadata(
            name=self.name,
            fullname=self.fullname,
            description=self.description,
            goal=self.goal,
        )

    @metadata.setter
    def metadata(self, metadata: truffle.AppMetadata):
        self.__dict__.update(metadata.model_dump())
    
    def tool(self, name: str=None, description: str=None, start_fn: Callable=None, stop_fn: Callable=None):
        '''
        Make a function into a tool. Use like a decorator or a function call.
        
        Example as a decorator:
        >>> app = App(name="test", fullname="Test App", description="Test app", goal="Testing")
        >>> @app.tool
        ... def my_tool(arg1, arg2):
        ...     return f"{arg1} {arg2}"
        >>> my_tool(1, 2)  # works like my_tool would normally as a function
        '1 2'
        
        Example showing a tool with start and stop functions:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> import time
        >>> thread_pool = ThreadPoolExecutor(max_workers=4)
        >>> def heavy_computation(x): 
        ...     time.sleep(1)  # Simulate delay
        ...     result = x * 2
        ...     print(f"Computed {x} * 2 = {result}")
        ...     return result
        
        >>> @app.tool
        ... def process_data(data_array):
        ...     """Process array of data in background thread pool"""
        ...     futures = [thread_pool.submit(heavy_computation, x) for x in data_array]
        ...     return [f.result() for f in futures]
        >>> process_data([1, 2, 3, 4])
        Computed 1 * 2 = 2
        Computed 2 * 2 = 4
        Computed 3 * 2 = 6
        Computed 4 * 2 = 8
        [2, 4, 6, 8]
        
        >>> @process_data.on_stop
        ... def cleanup_threads():
        ...     """Clean up thread pool when stopping"""
        ...     thread_pool.shutdown(wait=True)
        >>> thread_pool.shutdown()  # cleanup for doctest
        
        >>> tool = app.tool(name="process_data", description="Process array of data in background thread pool")
        >>> tool.start(list(range(10)))
        >>> time.sleep(3.5)
        >>> tool.stop()
        Computed 0 * 2 = 0
        Computed 1 * 2 = 2
        Computed 2 * 2 = 4
        
        Example as a function call:
        >>> tool = app.tool(start_fn=process_data, stop_fn=cleanup_threads)
        >>> tool.start(list(range(10)))
        >>> time.sleep(3.5)
        >>> tool.stop()
        '''
        def decorator(_start_fn: Callable):
            _name = name or _start_fn.__name__
            _description = description or _start_fn.__doc__ or str(inspect.signature(_start_fn))
            tool = Tool(name=_name, description=_description, start=_start_fn, stop=stop_fn)
            self.tools.append(tool)
            return tool
        
        if start_fn is not None:
            return decorator(start_fn)
        return decorator

    def start(self):
        for tool in self.tools:
            tool.register_with_truffle()
        
        # the naming is outdated but this is just starting a grpc server that exposes the endpoints we decorated with @app.tool
        return truffle.build_app_from_class(self)
