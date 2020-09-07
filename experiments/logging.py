import dataclasses
import json
from datetime import datetime
from os import PathLike
from typing import ClassVar, Optional, TextIO

from experiments.base import Logger


@dataclasses.dataclass
class JSONLogger(Logger):
    filepath: PathLike
    file: Optional[TextIO] = dataclasses.field(init=False)
    index: int = dataclasses.field(default=0, init=False)
    write_index: int = dataclasses.field(default=0, init=False)

    BUFFERING: ClassVar[int] = 1
    OPEN_MODE: ClassVar[str] = "wx"
    INDEX_KEY: str = "id"
    WRITE_INDEX_KEY: str = "uid"
    TIMESTAMP_KEY: str = "ts"
    DATA_KEY: str = "data"

    def append(self, key, value):
        log = {
            self.INDEX_KEY: key,
            self.WRITE_INDEX_KEY: self.write_index,
            self.TIMESTAMP_KEY: str(datetime.utcnow()),
            self.DATA_KEY: value,
        }
        self.file.write(f"{json.dumps(log)}\n")
        self.write_index += 1

    def push(self, value):
        self.append(self.index, value)
        self.index += 1

    def __enter__(self):
        self.file = open(self.filepath, self.OPEN_MODE, buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
