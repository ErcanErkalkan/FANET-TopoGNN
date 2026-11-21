from __future__ import annotations
from types import SimpleNamespace
import unittest
import numpy as np
from fanet.r4_evidence import restrict_to_frozen_temporal_support


def snap(run,t):
    return SimpleNamespace(run_id=run,time_index=t)

class CommonTemporalSupportTests(unittest.TestCase):
    def setUp(self):
        self.full=[snap('a',t) for t in range(12)]+[snap('b',t) for t in range(10)]

    def test_static_and_temporal_reduce_to_same_fixed_support(self):
        full=self.full
        temporal=[s for s in full if s.time_index>=4]
        static_result=restrict_to_frozen_temporal_support(full,full,np.arange(22),np.arange(22),window=5)
        temporal_result=restrict_to_frozen_temporal_support(full,temporal,np.arange(14),np.arange(14),window=5)
        self.assertEqual([(s.run_id,s.time_index) for s in static_result[0]],
                         [(s.run_id,s.time_index) for s in temporal_result[0]])
        self.assertEqual(len(static_result[0]),14)

    def test_missing_one_required_key_fails_closed(self):
        temporal=[s for s in self.full if s.time_index>=4][1:]
        with self.assertRaisesRegex(ValueError,'Missing common temporal prediction'):
            restrict_to_frozen_temporal_support(self.full,temporal,np.ones(len(temporal)),np.ones(len(temporal)))

    def test_unexpected_partial_extra_prediction_fails(self):
        temporal=[s for s in self.full if s.time_index>=4]
        weird=[self.full[0]]+temporal
        with self.assertRaisesRegex(ValueError,'Unexpected partial'):
            restrict_to_frozen_temporal_support(self.full,weird,np.ones(len(weird)),np.ones(len(weird)))

    def test_duplicate_key_and_nan_fail(self):
        full=self.full
        with self.assertRaisesRegex(ValueError,'Duplicate aligned'):
            restrict_to_frozen_temporal_support(full,full+[full[0]],np.ones(23),np.ones(23))
        with self.assertRaisesRegex(ValueError,'NaN/inf'):
            a=np.ones(22);a[0]=np.nan
            restrict_to_frozen_temporal_support(full,full,a,np.ones(22))

if __name__=='__main__':unittest.main()
