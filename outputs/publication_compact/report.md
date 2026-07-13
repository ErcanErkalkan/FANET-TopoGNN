# publication_compact report

- Seeds: 3
- Forecast horizon steps: 6
- Best model by MAE: Current-state persistence baseline
- Best model by lead: FANET-TopoGNN, FANET-TopoGNN (concat), GAT, GCN, Shallow ML, GraphSAGE, PI+MLP, STGCN (w=5)
- Wall-clock runtime: 4017.1 s
- CUDA available during run: false

## Accuracy
- Best MAE model: Current-state persistence baseline
- MAE mean: 0.1195
- R2 mean: 0.4249

## Early warning
- Best median lead model: FANET-TopoGNN, FANET-TopoGNN (concat), GAT, GCN, Shallow ML, GraphSAGE, PI+MLP, STGCN (w=5)
- Median lead mean: 600.00 ms

## Risk detection
                             Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
                 Kinetic-TopoGuard      0.736288          0.886560             0.636430
                         TGN (w=5)      0.366579          0.957539             0.227499
                            PI+MLP      0.317687          0.991277             0.195858
                       T-GCN (w=5)      0.331626          0.965839             0.203279
            FANET-TopoGNN (concat)      0.325112          0.997674             0.199128
                     FANET-TopoGNN      0.318906          0.998275             0.192956
                        Shallow ML      0.280723          1.000000             0.164818
                       STGCN (w=5)      0.338217          0.966600             0.205428
                               GCN      0.318279          0.966069             0.192314
                               GAT      0.316984          0.975148             0.192799
                         GraphSAGE      0.328537          0.983257             0.200928
Current-state persistence baseline      0.761909          0.770568             0.753497

## Network impact
                 Model  Connectivity ratio_mean  Reachability-delivery proxy (%)_mean  Proxy delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
     Kinetic-TopoGuard                 0.905787                             99.685707              18.715794                    1.948570               0.555296         2516.000000
                   GAT                 0.904676                             99.692112              18.488520                   25.847347               0.577830         6487.000000
                   GCN                 0.904306                             99.688941              18.499107                   25.101907               0.576938         6424.333333
         FANET-TopoGNN                 0.903843                             99.690444              18.482981                   24.999129               0.579975         6519.666667
             GraphSAGE                 0.903704                             99.690350              18.518053                   22.134411               0.581271         6265.000000
FANET-TopoGNN (concat)                 0.903472                             99.689241              18.489448                   24.094424               0.581509         6451.666667
