from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ...services import dataset_service as ds

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/summary")
def summary():
    return ds.dataset_summary()


@router.get("/quality-report")
def quality_report():
    return JSONResponse(ds.dataset_summary(), headers={"Content-Disposition": "attachment; filename=data_quality_report.json"})
