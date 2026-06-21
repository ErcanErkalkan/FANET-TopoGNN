# paper_like_submission report

- Seeds: 20
- Forecast horizon steps: 6
- Best model by MAE: Kinetic-TopoGuard
- Best model by lead: Shallow ML
- Wall-clock runtime: 11700.4 s
- CUDA available during run: false

## Accuracy
- Best MAE model: Kinetic-TopoGuard
- MAE mean: 0.3077
- R2 mean: 0.5391

## Early warning
- Best median lead model: Shallow ML
- Median lead mean: 495.00 ms

## Risk detection
                      Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
Union-Find detection oracle      0.754459          0.760507             0.748520
          Kinetic-TopoGuard      0.587066          0.995549             0.418609
                 Shallow ML      0.503403          0.975555             0.343349

## Network impact
            Model  Connectivity ratio_mean  PDR (%)_mean  Avg. end-to-end delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
Kinetic-TopoGuard                 0.841153     99.331567                        21.499666                   18.362444               1.251953             6589.85
