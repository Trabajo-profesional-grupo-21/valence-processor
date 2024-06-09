from pydantic_settings import BaseSettings, validator

class Settings(BaseSettings):

    VALENCE_MODEL: str | None = 'PYFEAT'

    REMOTE_RABBIT: bool | None = False
    RABBIT_HOST: str | None = None
    RABBIT_PORT: int | None = None
    RABBIT_VHOST: str | None = None
    RABBIT_USER: str | None = None
    RABBIT_PASSWORD: str | None = None

    @validator('VALENCE_MODEL')
    def validate_valence_model(cls, value):
        if value not in {'PYFEAT', 'FERPLUS'}:
            raise ValueError("VALENCE_MODEL must be either 'PYFEAT' or 'FERPLUS'")
        return value

    class Config:
        case_sensitive = True
        env_file = '.env'

settings = Settings()