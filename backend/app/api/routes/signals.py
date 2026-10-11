from fastapi import APIRouter, HTTPException, Query
from app.schemas.signal import RiskSignal

router = APIRouter(prefix="/signals", tags=["Signals"])

# Temporary storage; we will replace this with SQLite.
signals: dict[str, RiskSignal] = {}


@router.post("", response_model=RiskSignal, status_code=201)
def create_signal(signal: RiskSignal):
    if signal.signal_id in signals:
        raise HTTPException(status_code=409, detail="Signal already exists")

    signals[signal.signal_id] = signal
    return signal


@router.get("", response_model=list[RiskSignal])
def get_signals(limit: int = Query(default=50, ge=1, le=200)):
    return list(signals.values())[::-1][:limit]


@router.get("/{signal_id}", response_model=RiskSignal)
def get_signal(signal_id: str):
    signal = signals.get(signal_id)

    if signal is None:
        raise HTTPException(status_code=404, detail="Signal not found")

    return signal