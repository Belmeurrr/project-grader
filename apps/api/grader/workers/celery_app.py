from celery import Celery

from grader.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "grader",
    broker=settings.celery_broker_url,
    backend=settings.celery_broker_url,
    include=["grader.workers.grading_pipeline"],
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="grading",
    task_routes={
        "grader.workers.grading_pipeline.*": {"queue": "grading"},
    },
)
