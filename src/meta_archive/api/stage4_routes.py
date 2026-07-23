from fastapi import APIRouter

from meta_archive.api.activity_routes import activity_router
from meta_archive.api.analytics_routes import analytics_router
from meta_archive.api.event_routes import event_router
from meta_archive.api.people_routes import people_router

stage4_router = APIRouter()
stage4_router.include_router(activity_router)
stage4_router.include_router(event_router)
stage4_router.include_router(people_router)
stage4_router.include_router(analytics_router)
