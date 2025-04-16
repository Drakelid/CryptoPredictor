from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from typing import List, Optional
import logging
import pandas as pd
import io

from app.schemas.data import DataFetchInput, DataUploadInput, DataInfo
from app.services.data_service import DataService

router = APIRouter()
logger = logging.getLogger(__name__)

# Create service instance
data_service = DataService()

@router.post("/fetch", response_model=DataInfo)
async def fetch_data(
    input_data: DataFetchInput
):
    """
    Fetch cryptocurrency data from external source
    """
    try:
        result = data_service.fetch_data(
            symbol=input_data.symbol,
            source=input_data.source,
            days=input_data.days
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=DataInfo)
async def upload_data(
    input_data: DataUploadInput = Depends(),
    file: UploadFile = File(...)
):
    """
    Upload cryptocurrency data from CSV file
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Save data
        result = data_service.save_data(
            df=df,
            symbol=input_data.symbol,
            source=input_data.source
        )

        return result
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info", response_model=List[DataInfo])
async def get_data_info(
    symbol: Optional[str] = Query(None, description="Cryptocurrency symbol (e.g., BTC, ETH)")
):
    """
    Get information about available data
    """
    try:
        results = data_service.get_data_info(symbol=symbol)
        return results
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview/{symbol}/{source}")
async def preview_data(
    symbol: str,
    source: str,
    limit: int = Query(100, description="Number of rows to return")
):
    """
    Preview cryptocurrency data
    """
    try:
        df = data_service.load_data(symbol=symbol, source=source)

        # Convert to dict for JSON response
        result = df.head(limit).to_dict(orient='records')

        return result
    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
