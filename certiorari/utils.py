import inspect
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, Self

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

    def tool(self, tool: Callable):
        truffle.register_tool(tool)
        self.tools.append(tool)
        assert tool.__name__ is not None
        assert not hasattr(self, tool.__name__)
        setattr(self, tool.__name__, tool)

    def start(self):

        truffle_app = truffle.build_app_from_class(self)

        # Inject metadata and app into caller's namespace and __main__ because the truffle sdk
        # seems to have no cleaner way to make a app run

        frame = inspect.currentframe().f_back
        first_frame = True
        while frame:
            frame = frame.f_back
            found_main_frame = frame.f_globals["__name__"] == "__main__"
            found_first_calling_frame = first_frame

            if found_first_calling_frame or found_main_frame:
                frame.f_locals["app"] = truffle_app

            if found_first_calling_frame:
                first_frame = False
            if found_main_frame:
                break

        return truffle_app
