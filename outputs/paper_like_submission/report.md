# paper_like_submission report

- Seeds: 20
- Forecast horizon steps: 6
- Best model by MAE: Current-state persistence baseline
- Best model by lead: Shallow ML
- Wall-clock runtime: 45.3 s
- CUDA available during run: false

## Accuracy
- Best MAE model: Current-state persistence baseline
- MAE mean: 0.1319
- R2 mean: 0.4493

## Early warning
- Best median lead model: Shallow ML
- Median lead mean: 595.00 ms

## Risk detection
                             Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
                 Kinetic-TopoGuard      0.727802          0.870056             0.634079
                        Shallow ML      0.353488          0.975079             0.223733
Current-state persistence baseline      0.758144          0.766140             0.750380

## Network impact
            Model  Connectivity ratio_mean  Reachability-delivery proxy (%)_mean  Proxy delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
Kinetic-TopoGuard                 0.894653                             99.653137              18.596993                    2.789627               0.564636             2773.35
