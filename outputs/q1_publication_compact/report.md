# q1_publication_compact report

- Seeds: 3
- Forecast horizon steps: 6
- Best model by MAE: Union-Find detection oracle
- Best model by lead: FANET-TopoGNN, PI+MLP

## Accuracy
- Best MAE model: Union-Find detection oracle
- MAE mean: 0.3045
- R2 mean: 0.3776

## Early warning
- Best median lead model: FANET-TopoGNN, PI+MLP
- Median lead mean: 600.00 ms

## Risk detection
                      Model  Risk_F1_mean  Risk_Recall_mean  Risk_Precision_mean
Union-Find detection oracle      0.731125          0.737960             0.724422
          Kinetic-TopoGuard      0.554739          0.996787             0.385514
     FANET-TopoGNN (concat)      0.518727          0.971668             0.355212
                STGCN (w=5)      0.507992          0.980429             0.344025
                        GCN      0.501973          0.975217             0.339958
                  TGN (w=5)      0.494184          0.974348             0.335781
                  GraphSAGE      0.486667          0.992282             0.323793
                        GAT      0.468494          0.986288             0.308933
              FANET-TopoGNN      0.445748          0.998216             0.288750
                T-GCN (w=5)      0.444925          0.995728             0.287361
                 Shallow ML      0.443251          0.972199             0.287274
                     PI+MLP      0.415762          0.991493             0.265647

## Network impact
                 Model  Connectivity ratio_mean  PDR (%)_mean  Avg. end-to-end delay (ms)_mean  Proactive reroute (%)_mean  DTN buffered (%)_mean  Relay actions_mean
             GraphSAGE                 0.861296     99.387386                        21.458443                   22.360033               1.145574         7129.333333
     Kinetic-TopoGuard                 0.861296     99.387386                        21.504447                   18.065609               1.145574         6282.333333
                   GAT                 0.861250     99.387199                        21.441894                   26.717662               1.145574         7479.333333
                   GCN                 0.861111     99.386432                        21.445651                   23.016687               1.145574         6830.666667
         FANET-TopoGNN                 0.861019     99.384608                        21.423077                   27.863993               1.145574         7847.333333
FANET-TopoGNN (concat)                 0.860139     99.374181                        21.458397                   20.618180               1.144538         6553.333333
