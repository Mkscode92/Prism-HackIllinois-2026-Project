from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Anthropic, for AI-gen fixes after a review is posted
    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-sonnet-4-6", validation_alias="CLAUDE_MODEL")

    # Voyage AI
    voyage_api_key: str = Field(..., validation_alias="VOYAGE_API_KEY")
    voyage_model: str = Field("voyage-code-2", validation_alias="VOYAGE_MODEL")

    # Pinecone
    pinecone_api_key: str = Field(..., validation_alias="PINECONE_API_KEY")
    pinecone_index: str = Field("prism-code-index", validation_alias="PINECONE_INDEX")
    pinecone_top_k: int = Field(10, validation_alias="PINECONE_TOP_K")

    # GitHub
    github_token: str = Field(..., validation_alias="GITHUB_TOKEN")

    # Modal
    modal_app_name: str = Field("prism-sandbox", validation_alias="MODAL_APP_NAME")

    # Poller
    poll_interval_seconds: int = Field(900, validation_alias="POLL_INTERVAL_SECONDS")
    max_pending_reviews: int = Field(10, validation_alias="MAX_PENDING_REVIEWS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore"
    }


settings = Settings()
