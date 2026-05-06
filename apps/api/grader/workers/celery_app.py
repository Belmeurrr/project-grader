from celery import Celery
from celery.schedules import crontab  # noqa: F401  (kept for future cron-style entries)

from grader.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "grader",
    broker=settings.celery_broker_url,
    backend=settings.celery_broker_url,
    include=[
        "grader.workers.grading_pipeline",
        "grader.workers.reconciler",
    ],
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="grading",
    task_routes={
        "grader.workers.grading_pipeline.*": {"queue": "grading"},
        # Reconciler is cheap (one indexed range scan + a few row updates)
        # and must not get queued behind heavy ML pipeline tasks. Pin it
        # to its own queue so a saturated `grading` queue doesn't keep
        # orphans wedged for hours.
        "grader.workers.reconciler.*": {"queue": "maintenance"},
    },
)

# Beat schedule. We embed beat in the worker process for now (single
# `celery -A grader.workers.celery_app worker -B`) — splitting beat into
# a separate deployment is a Phase-2 concern. Run cadence is 2 minutes:
# tight enough that a wedged submission flips to FAILED within ~7 min
# of the worker death (5min stale window + up to 2min until next tick),
# loose enough that the lock-skipping SELECT is cheap.
celery_app.conf.beat_schedule = {
    "reconcile-stale-submissions": {
        "task": "grader.workers.reconciler.reconcile_stale_submissions",
        "schedule": 120.0,  # seconds
        "options": {"queue": "maintenance", "expires": 110.0},
    },
}
