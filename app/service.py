"""
U-CS 41 — your logic goes here.

STEP 1 IS THE BASELINE. The dumbest thing that works, with a number you have measured.
Do not replace it with a model until that number exists and you know what to beat.
"""
from app.evaluation import baseline_probability
from app.schemas import Request, Response


def handle(body: Request) -> Response:
    p = baseline_probability()
    return Response(
        probability=p,
        factors=[f"baseline: same probability for every booking ({p:.1%} training-set no-show rate)"],
        method="baseline: constant training-set no-show rate",
    )
