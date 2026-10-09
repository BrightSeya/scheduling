"""U-CS 41 no-show prediction: look up the booking, score it, explain it. Data is SYNTHETIC."""
from functools import lru_cache

from app.explain import factors
from app.features import features_for_booking
from app.model import TrainedModel, train
from app.schemas import Request, Response


@lru_cache(maxsize=1)
def get_model() -> TrainedModel:
    return train()


def warm_up() -> None:
    get_model()


def handle(body: Request) -> Response:
    """Raises UnknownBooking if the bookingId is not in the dataset."""
    features = features_for_booking(body.bookingId)
    model = get_model()
    return Response(
        probability=float(model.probability(features)[0]),
        factors=factors(model.pipeline, features),
        method="logistic regression, Platt-calibrated",
    )
