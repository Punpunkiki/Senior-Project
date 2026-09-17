"""app/config.py -- runtime settings for the "หมอทุเรียน" LINE OA service.

Loaded from environment variables / .env (see .env.example). This is
separate from config.yaml, which stays the single source of truth for the
ML pipeline (architectures, classes, transforms). Nothing here is a secret
default -- LINE tokens must come from the environment.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LINE Messaging API -------------------------------------------------
    line_channel_secret: str = ""
    line_channel_access_token: str = ""

    # --- LIFF -----------------------------------------------------------------
    liff_id: str = ""
    # Channel ID of the LIFF app's provider channel. Required to verify that
    # the viewer of a result page is the farmer who sent the photo; when it is
    # empty the ownership check cannot run (see app/auth.py).
    liff_channel_id: str = ""

    # --- Model serving ----------------------------------------------------
    # Points at the training pipeline's own config.yaml -- classes, image
    # size, normalization all come from there, never duplicated here.
    config_yaml_path: str = str(REPO_ROOT / "config.yaml")
    # Which of the 3 trained backbones to serve (name must match
    # config.yaml -> models[].name and have outputs/<model_name>/best.pt).
    model_name: str = "efficientnet_b0"

    # --- Confidence tiers ---------------------------------------------------
    # Fixed UX cutoffs (product decision), independent of the training
    # pipeline's own calibrated precision/coverage threshold search in
    # src/evaluate.py. See app/diagnosis.py.
    confidence_high: float = 0.75
    confidence_low: float = 0.45

    # --- Storage --------------------------------------------------------------
    database_url: str = "sqlite:///./durian_bot.db"
    upload_dir: str = "./uploads"
    image_retention_days: int = 30
    # LINE itself caps image messages at 10 MB; this is our own guard so a
    # huge upload cannot exhaust memory during preprocessing.
    max_image_mb: float = 10.0

    # --- Public URLs (fill in once deployed; see line-assets/HANDOFF.md) ---
    base_url: str = "http://localhost:8000"
    web_base_url: str = "http://localhost:3000"

    # --- Human fallback -------------------------------------------------------
    admin_contact_text: str = (
        "ติดต่อเจ้าหน้าที่เกษตรได้ที่เบอร์ 02-XXX-XXXX "
        "(จันทร์-ศุกร์ 08:30-16:30)"
    )

    port: int = 8000


settings = Settings()
