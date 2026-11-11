# FRE-GY 9743 - Assignment 7

**Aditya Amit Shah | as22008**

The [executed notebook](hw7_monotone_convex_interpolator.ipynb) contains the
method, proofs, IFR plots, all six original pricing examples, results and
interpretation. The [HTML copy](submission/Assignment_7_as22008.html) includes
the saved outputs for reading without Python (math rendering may require an
internet connection).

## Method

`Interpolator1DMonotoneConvex` in `fixedincomelib/utilities/numerics.py`
implements a custom piecewise-quadratic IFR **point** interpolator with exact
polynomial integration and flat extrapolation. It is not an implementation
of Hagan-West's interval-average-forward algorithm.

For compatible data, the method is continuous, interpolates every point,
preserves monotonicity, and is convex throughout the node interval, including
across knots. Curvature is bounded using adjacent secant-slope differences.
It need not have continuous first derivatives. The notebook gives the full
construction and proof. A one-node curve is constant everywhere.

For incompatible data, an explicit warning and diagnostic attributes identify
a piecewise-linear fallback that preserves the supplied quotes. It does not
claim global monotonicity and convexity when the point data make them
impossible. Invalid nodes are rejected. The two optional risk methods remain
as the instructor's stubs.

## Supplied-data results

Value date: 2026-01-02. Notional: USD 1,000,000.

| Product | PCP PV (USD) | Quadratic PV (USD) | Change (USD) |
| --- | ---: | ---: | ---: |
| FRA, receive fixed | 2,908.286639 | 3,263.971122 | +355.684483 |
| RFR swap, receive fixed | 6,222.517513 | 12,836.477674 | +6,613.960161 |
| Basis swap, receive SOFR + spread | 8,351.348908 | 7,622.813583 | -728.535325 |

The quadratic curve lies below the PCP right-endpoint rectangles for these
increasing inputs. It lowers integrated forward rates and raises discount
factors. Receiving fixed benefits from lower floating projections. The basis
swap falls because the received SOFR leg declines more than the paid Fed
Funds leg; the notebook separately reports each signed leg's contribution.

All original data, product conventions and pricing API calls are retained.
Only the interpolation-method setting differs within each pair. No calibration
is performed. The constant 5 bp funding spread is added to the SOFR reference
curve under both methods.

## Run

The instructor's pinned NumPy requirements target Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute_notebook.py
python -m unittest discover -s tests -v
```

Run commands from the repository root so the static convention files are
available. Alternatively, open the notebook in Jupyter, restart the kernel
and run all cells. Select the environment containing the required packages.

The execution script runs every cell in a fresh IPython process and saves the
notebook plus HTML. It avoids opening kernel sockets, so it also works in
restricted environments. The ordinary notebook cells do not depend on it.

Local verification used Python 3.13.6, NumPy 2.3.2, pandas 2.3.1,
QuantLib 1.41, matplotlib 3.10.5 and nbconvert 7.16.6. This uses the available
local packages rather than the starter's Python 3.11/NumPy 1.26.4 environment;
that pinned combination was not independently run in this session.

## Validation

All **18 automated tests** pass, and all **11 notebook code cells** executed
successfully in a fresh IPython process with saved outputs.

- Exact known quadratic solutions and their analytic integrals.
- Independent Gauss-Legendre quadrature, split at knots, for arbitrary limits.
- Continuity, no overshoot, monotonicity and convexity, including 120 seeded
  random compatible curves and arbitrary Jensen inequalities.
- Constant, two-node, irregular-grid, negative-rate and flat-tail cases.
- Disclosed fallback, invalid data, unchanged inputs and optional-risk stubs.
- All six prices reconstructed from cashflows and numerical IFR integration,
  without the model discount-factor or interpolator-integral methods.
- Funding-spread handling and unchanged convention dictionaries.

The notebook also reconciles all six signed cashflow reports to their PVs.
The exported numerical tables and plot are in `submission/`.

## Files to submit

The main deliverable is the complete project: notebook, `fixedincomelib/`,
`static_files/`, requirements and supporting outputs. A notebook alone cannot
run without the library. Use the complete ZIP when file upload is required,
and include the HTML copy for easy review if the portal allows it.

The local solution branch is `Aditya_Shah_Branch`. Its configured push
destination is the personal fork:

`https://github.com/Aditya8321/FRE-GY-9743-Assignments-7.git`

After publishing, its branch URL is:

`https://github.com/Aditya8321/FRE-GY-9743-Assignments-7/tree/Aditya_Shah_Branch`

The announcement gives **November 12, 2026** as the deadline but does not
specify a time or submission format. Follow the actual Brightspace fields.
A GitHub push is separate from submitting the assignment through Brightspace.

Instructor starter: https://github.com/Jerry0108-SJ/FRE-GY-9743-Assignments-7
