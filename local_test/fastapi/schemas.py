from pydantic import BaseModel

class SettingsUpdate(BaseModel):
    user_id: str
    type: str
    value: str