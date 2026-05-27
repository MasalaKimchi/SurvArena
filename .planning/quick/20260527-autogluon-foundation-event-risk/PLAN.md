---
status: in_progress
created: 2026-05-27
---

# AutoGluon Foundation Event-Risk Adapters

Implement AutoGluon event-risk survival adapters for non-Mitra tabular foundation models, add method configs, remove Mitra from no-HPO manuscript configs, and attempt manuscript no-HPO runs for clinical and genomics tracks.

## Plan

1. Add thin AutoGluon event-risk adapter subclasses for TabICL, TabM, TabDPT, and AutoGluon RealTabPFN-V2.
2. Register the adapters, catalog them as implemented foundation methods, and keep native-categorical AutoGluon preprocessing.
3. Add method YAML configs and update no-HPO manuscript clinical/genomics configs to include the new adapters while excluding Mitra.
4. Add focused unit coverage for registration, hyperparameter wiring, and benchmark config membership.
5. Run touched tests and then attempt the two manuscript no-HPO benchmark commands.
