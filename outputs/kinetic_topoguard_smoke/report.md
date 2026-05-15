# kinetic_topoguard_smoke report

- Seeds: 1
- Forecast horizon steps: 6
- Best model by MAE: Kinetic-TopoGuard
- Best model by lead: Kinetic-TopoGuard, Shallow ML

## Accuracy
- Best MAE model: Kinetic-TopoGuard
- MAE mean: 0.3858
- R2 mean: 0.4403

## Early warning
- Best median lead model: Kinetic-TopoGuard
- Median lead mean: 400.00 ms

## Risk detection
                      Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
Union-Find detection oracle      0.740659          0.745575             0.735808
          Kinetic-TopoGuard      0.600266          1.000000             0.428843
                 Shallow ML      0.590551          0.995575             0.419776

## Network impact
            Model  Connectivity ratio_mean  PDR (%)_mean  Avg. end-to-end delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
Kinetic-TopoGuard                 0.981944     99.975746                        16.443323                   19.359142               0.048507              1512.0
