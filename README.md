# verify_robustness
Using 'auto_LiRPA' to test our pretrained model (a single model and an ensemble model)

### Requirements
python (3.0+)

pytorch (0.4.1+)

First, you should train a model.

If you train a single model, you can run `python train_single.py`

If you train an ensemble model, you can run `python train_ensemble.py`

When you get the pretrained model, you can verify the robustness of pretrained model.

If you verify the single model, you can run `python ensemble_verification.py`

If you verify the ensemble model, you can run `python train_ensemble.py`
