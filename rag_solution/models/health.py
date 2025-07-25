
from enum import Enum
from pydantic import BaseModel

class StatusEnum(str, Enum):
    OK = "ok"
    ERROR = "error"

    @classmethod
    def from_bool(cls, value: bool) -> "StatusEnum":
        return cls.OK if value else cls.ERROR

class HealthCheckResponse(BaseModel):
    db: StatusEnum
    worker_pool: StatusEnum

    def get_response_code(self) -> int:
        if self.db == StatusEnum.OK and self.worker_pool == StatusEnum.OK:
            return 200
        return 503
