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

    # def tool(self, tool: Callable):
    #     # TODO: we can currently do this
    #     # @app.tool
    #     # def my_tool(arg1, arg2):
    #     #     ...
    #     # but we want to make this the default start method and then make it so that you can also decorate a stop method
    #     # @app.tool
    #     # def my_tool(arg1, arg2):
    #     #     ...
    #     # my_tool()

    #     @functools.wraps(tool)
    #     def wrapper(*args, **kwargs):
    #         return tool(*args, **kwargs)

    #     truffle.register_tool(wrapper)
    #     self.tools.append(wrapper)
    #     assert tool.__name__ is not None
    #     assert not hasattr(self, tool.__name__)
    #     setattr(self, tool.__name__, wrapper)
    
    def tool(self, name: str=None, description: str=None, start_fn: Callable=None, stop_fn: Callable=None):
        '''
        Decorate a function to be a tool.
        
        @app.tool
        def my_tool(arg1, arg2):
            ...
        my_tool() # works like my_tool would normally
        '''
        def decorator(_start_fn: Callable):
            name = name or _start_fn.__name__
            description = description or _start_fn.__doc__ or str(inspect.signature(_start_fn))
            tool = Tool(name=name, description=description, start=_start_fn)
            self.tools.append(tool)
            
            @functools.wraps(_start_fn)
            def wrapper(*args, **kwargs):
                return _start_fn(*args, **kwargs)
            
            @functools.wraps(tool.start)
            def _start_fn(*args, **kwargs):
                return tool.start(*args, **kwargs)
            
            wrapper.start = _start_fn
            
            def on_stop(_stop_fn: Callable):
                
                @functools.wraps(_stop_fn)
                def stop(*args, **kwargs):
                    return _stop_fn(*args, **kwargs)
                wrapper.stop = stop
            
            if stop_fn is not None:
                wrapper.on_stop(_stop_fn=stop_fn)
            
            return wrapper
    
        if start_fn is not None:
            return decorator(_start_fn=start_fn)
        return decorator

    def _register_tools(self):
        for tool in self.tools:
            if tool.start and tool.stop:
                truffle.register_tool(tool.start, name=f'start_{tool.name}')
                truffle.register_tool(tool.stop, name=f'stop_{tool.name}')
            elif tool.start:
                truffle.register_tool(tool.start, name=tool.name)
            else:
                raise ValueError(f'Tool {tool.name} must have a start method')

    def start(self):
        self._register_tools()
        
        # the naming is outdated but this is just starting a grpc server that exposes the endpoints we decorated with @app.tool
        return truffle.build_app_from_class(self)
