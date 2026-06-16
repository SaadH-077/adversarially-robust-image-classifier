import os
import sys
import requests

"""
Submission script for the Robustness task (TML Assignment 3).

Submission requirements (read carefully to avoid automatic rejection):
1. FILE FORMAT  - a PyTorch state_dict saved as .pt:
       torch.save(model.state_dict(), "model.pt")   # correct
       torch.save(model, "model.pt")                 # WRONG
2. ARCHITECTURE - model-name in {resnet18, resnet34, resnet50}; must match the state_dict.
3. MODEL        - input (1,3,32,32), output (1,9); only the final fc replaced for 9 classes.
4. EVALUATION   - clean accuracy must be > 50% or the submission is rejected.
                  score = 0.5 * clean_accuracy + 0.5 * robustness_accuracy
5. LIMITS       - one submission per group every 60 minutes (2 min cooldown on error).

Our winning submission: resnet34 + Friendly-MART + EMA + SGDR (state_dict produced by
train_robust_classifier.ipynb, public leaderboard score 0.642964). MODEL_NAME = "resnet34".
"""

BASE_URL = "http://34.63.153.158"

# >>> FILL THESE IN BEFORE SUBMITTING <<<
API_KEY    = "YOUR_API_KEY_HERE"                 # team API key (do NOT commit the real key)
MODEL_PATH = "PATH/TO/resnet34_r34_fatmart_tau2_data_submission.pt"  # the state_dict to submit
MODEL_NAME = "resnet34"                           # must match MODEL_PATH (resnet18/34/50)

SUBMIT = False  # set to True only when you actually want to push to the server

TASK_ID = "03-robustness"  # do not change


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


if SUBMIT:
    if API_KEY == "YOUR_API_KEY_HERE":
        die("Set API_KEY to your team key before submitting.")
    if not os.path.isfile(MODEL_PATH):
        die(f"File not found: {MODEL_PATH}")

    try:
        with open(MODEL_PATH, "rb") as f:
            files = {"file": (os.path.basename(MODEL_PATH), f, "application/x-pytorch")}
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                data={"model_name": MODEL_NAME},
            )

        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413).")

        resp.raise_for_status()
        print("Successfully submitted.")
        print("Server response:", body)

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
