# publication_neural_extension report

- Seeds: 2
- Forecast horizon steps: 6
- Best model by MAE: Kinetic-TopoGuard
- Best model by lead: FANET-TopoGNN, TGN (w=5), PI+MLP
- Wall-clock runtime: 2787.7 s
- CUDA available during run: false

## Accuracy
- Best MAE model: Kinetic-TopoGuard
- MAE mean: 0.3094
- R2 mean: 0.4473

## Early warning
- Best median lead model: FANET-TopoGNN, TGN (w=5), PI+MLP
- Median lead mean: 600.00 ms

## Risk detection
                 Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
     Kinetic-TopoGuard      0.567175          0.995936             0.396909
           STGCN (w=5)      0.493241          0.991517             0.328783
FANET-TopoGNN (concat)      0.483740          0.993936             0.322184
             GraphSAGE      0.477681          0.986506             0.315253
           T-GCN (w=5)      0.471542          0.996563             0.310945
                   GAT      0.460908          0.979725             0.302465
                   GCN      0.460537          0.980925             0.300981
         FANET-TopoGNN      0.444829          0.996419             0.286412
             TGN (w=5)      0.426632          0.999365             0.271582
                PI+MLP      0.392338          0.997727             0.244512

## Network impact
                 Model  Connectivity ratio_mean  PDR (%)_mean  Avg. end-to-end delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
         FANET-TopoGNN                 0.865278     99.497310                        21.116852                   28.482276               0.942895              7708.5
FANET-TopoGNN (concat)                 0.865278     99.497310                        21.155019                   22.977861               0.942895              7149.5
     Kinetic-TopoGuard                 0.865278     99.497310                        21.210843                   17.437189               0.942895              6051.5
             GraphSAGE                 0.865278     99.497310                        21.137049                   24.939335               0.942895              7131.5
                   GCN                 0.865208     99.497030                        21.124472                   27.169869               0.942895              7351.0
                   GAT                 0.865069     99.496471                        21.136172                   26.830162               0.942895              7418.0
