# FANET-TopoGNN — R2 strict-v3 experimental coverage (16 October 2026)

**Experimental checkpoint coverage: PASS (600/600). Manuscript release: HOLD.** This is a coverage/log audit, not a model-superiority claim.

A supplied Windows PowerShell execution log for Python 3.13.16 and PyTorch 2.14.1+cpu was independently parsed. It contains 20 distinct registered seeds, six horizon steps (2, 4, 6, 10, 15, 20), five model identifiers (persistence, extratrees, kinetic, source_gated, shallow), and **600 unique successful or verified-resumed model–seed–horizon keys**, with zero duplicated or missing keys.

The final runner audit states `validated_rows=600`, `missing_rows=0`, `full_family_cells=120`, `errors=[]`, `strict_r2_full_complete=true`. The v3-only independent coverage audit also reports 600 valid entries, 0 missing entries and `full_R2_20seed_6h_complete=true`. 22/22 unit tests passed in the user execution environment.

Caveats: numerical per-model metric JSONs were not included in the log upload, so seed-paired statistical model comparison and full source/environment verification remain OPEN. An earlier intermediate counter mismatch (447 versus 420) resolves to 600/600 at run completion; the final independent checker still prints an unconditional stale `warning: Incomplete full R2` string despite its final `true` completion flag. `model_sources_bitwise_equal_between_v2_v3=null` is not a failure because v2 was excluded from the homogeneous v3-only checker.

**Other release gates OPEN:** strict R3 event protocol; R4 14-family retraining; R5 policy dependence; complete code/data/license provenance; valid references, author disclosures and journal requirements. Preserve the frozen P6 manuscript and prior locked EAAI evidence until a separately controlled P8 candidate is justified. No field packet-delivery validation or model superiority is established here.
