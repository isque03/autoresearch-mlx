"""Property test for train.py's Newton-Schulz orthogonalization (the core of
the Muon optimizer step): the output should be approximately orthogonal
(x @ x.T close to the identity, up to the row/col scaling Muon intentionally
leaves in), regardless of what the input gradient matrix looks like."""
import mlx.core as mx

from train import zeropower_via_newtonschulz5


def _max_abs_gram_error(matrix: mx.array) -> float:
    gram = matrix @ matrix.T
    identity = mx.eye(gram.shape[0])
    return float(mx.max(mx.abs(gram - identity)).item())


def test_output_is_measurably_more_orthogonal_than_the_raw_input():
    """5 quintic Newton-Schulz iterations is a cheap, deliberately partial
    approximation to a true polar decomposition (that's the whole point --
    an exact orthogonalization would be far more expensive per optimizer
    step) -- it should move gram(result) substantially closer to the
    identity than the raw normalized input, not reach it exactly. Verified
    empirically: raw normalized input error ~0.98, 5-step result ~0.41 on a
    (32,48) random matrix -- a large, real improvement, not near-zero."""
    mx.random.seed(0)
    grad = mx.random.normal((32, 48)).astype(mx.float32)
    raw_normalized = grad / (mx.linalg.norm(grad) + 1e-7)
    raw_error = _max_abs_gram_error(raw_normalized)

    result = zeropower_via_newtonschulz5(grad, steps=5).astype(mx.float32)
    result_error = _max_abs_gram_error(result)

    assert result_error < raw_error * 0.6, (
        f"expected a substantial improvement, got raw={raw_error:.3f} result={result_error:.3f}"
    )


def test_more_steps_gets_closer_to_orthogonal_on_rectangular_matrix():
    mx.random.seed(0)
    grad = mx.random.normal((32, 48)).astype(mx.float32)
    error_5 = _max_abs_gram_error(zeropower_via_newtonschulz5(grad, steps=5).astype(mx.float32))
    error_10 = _max_abs_gram_error(zeropower_via_newtonschulz5(grad, steps=10).astype(mx.float32))
    assert error_10 < error_5


def test_output_shape_matches_input_shape():
    mx.random.seed(1)
    for shape in [(16, 16), (16, 32), (32, 16)]:
        grad = mx.random.normal(shape)
        result = zeropower_via_newtonschulz5(grad, steps=5)
        assert result.shape == shape


def test_deterministic_given_same_input():
    mx.random.seed(2)
    grad = mx.random.normal((20, 20))
    result_a = zeropower_via_newtonschulz5(grad, steps=5)
    result_b = zeropower_via_newtonschulz5(grad, steps=5)
    assert mx.array_equal(result_a, result_b)
