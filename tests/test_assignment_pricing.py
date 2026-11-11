"""Run the untouched pricing examples and verify with numerical IFR integration."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
from fixedincomelib.date import Date, accrued
from fixedincomelib.product import ProductOvernightIndexCashflow
from test_monotone_convex import quadrature


class AssignmentPricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'hw7_monotone_convex_interpolator.ipynb').read_text())
        cls.ns = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if set(cell.get('metadata', {}).get('tags', [])) & {'assignment-data', 'pricing-example'}:
                    exec(''.join(cell['source']), cls.ns)

    def numerical_integral(self, model, index, start, end):
        curve = model.retrieve_model_component(index).state_data_interpolator
        a, b = accrued(model.value_date, start), accrued(model.value_date, end)
        return quadrature(curve, a, b)

    def numerical_discount_factor(self, model, payment):
        integral = self.numerical_integral(model, 'SOFR-1B', model.value_date, payment)
        return np.exp(-integral - 0.0005*accrued(model.value_date, payment))

    def numerical_stream_pv(self, model, stream, index=None):
        total = 0.0
        for i in range(stream.num_cashflows()):
            cf = stream.cashflow(i)
            if stream.fixed_rate is not None:
                amount = cf.notional * cf.accrued * stream.fixed_rate
            else:
                # All assigned accruals are future, with business-day boundaries:
                # daily compound factors telescope to exp(integral f).
                integral = self.numerical_integral(model, index, cf.effective_date, cf.termination_date)
                amount = cf.notional * np.expm1(integral)
                if isinstance(cf, ProductOvernightIndexCashflow):
                    accrual = cf.on_index.dayCounter().yearFraction(cf.effective_date, cf.termination_date)
                    amount += cf.notional * cf.spread * accrual
            total += amount * self.numerical_discount_factor(model, cf.payment_date)
        return total

    def test_all_six_prices_against_independent_quadrature_cashflows(self):
        for prefix in ['fra', 'swap', 'basis']:
            for method in ['pcp', 'student']:
                with self.subTest(product=prefix,method=method):
                    model = self.ns[f'{prefix}_{method}_model']
                    product = self.ns[f'{prefix}_{method}_product']
                    if prefix == 'basis':
                        expected = (self.numerical_stream_pv(model, product.floating_leg_1, 'SOFR-1B')
                                    - self.numerical_stream_pv(model, product.floating_leg_2, 'FF-1B'))
                    else:
                        expected = (self.numerical_stream_pv(model, product.fixed_leg)
                                    - self.numerical_stream_pv(model, product.floating_leg, 'SOFR-1B'))
                    self.assertAlmostEqual(self.ns[f'{prefix}_{method}_pv'], expected, delta=1e-6)

    def test_only_interpolation_setting_changes(self):
        for prefix in ['fra','swap','basis']:
            a,b = self.ns[f'{prefix}_pcp_convention'], self.ns[f'{prefix}_student_convention']
            self.assertEqual(a.keys(), b.keys())
            self.assertEqual([k for k in a if a[k] != b[k]], ['interpolation method'])
            self.assertEqual(a['extrapolation method'], 'FLAT')

    def test_funding_spread_and_model_discount_factors(self):
        for method in ['pcp','student']:
            model = self.ns[f'fra_{method}_model']
            spread = model.retrieve_model_component('SOFR-1B-FLAT').state_data_interpolator
            for t in [0,0.1,1,3,8]:
                self.assertEqual(spread.interpolate(t), 0.0005)
                self.assertAlmostEqual(spread.integrate(0,t), 0.0005*t, places=14)
            for d in ['2026-02-02','2026-11-05','2028-07-03','2033-01-03']:
                self.assertAlmostEqual(model.discount_factor('SOFR-1B-FLAT', Date(d)),
                                       self.numerical_discount_factor(model, Date(d)), places=13)

    def test_supplied_curves_are_compatible_and_reproduce_quotes(self):
        model = self.ns['fra_student_model']
        for index, quotes in [('SOFR-1B','sofr_quotes'), ('FF-1B','ff_quotes')]:
            curve = model.retrieve_model_component(index).state_data_interpolator
            self.assertTrue(curve.shape_compatible)
            self.assertIsNone(curve.fallback_reason)
            for d, value in self.ns[quotes]['values'].items():
                t = accrued(model.value_date,Date(d))
                self.assertEqual(curve.interpolate(t), value)


if __name__ == '__main__':
    unittest.main()
