from pydantic_settings import BaseSettings


class InferenceConfig(BaseSettings):
    IMAGE_PROCESSING_WORKERS: int = 4

config = InferenceConfig()