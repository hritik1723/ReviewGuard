"""Hidden Markov Model (HMM) sequence analysis for ReviewGuard.

Classical techniques:
1. Two-state HMM (State 0: Normal organic reviewer, State 1: Spam / burst reviewer)
2. Forward algorithm for sequence log-likelihood: log P(O | lambda)
3. Backward algorithm for posterior state probabilities
4. Viterbi decoding algorithm for the most likely hidden state trajectory
5. Sequence-based spam risk scoring per user
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .load_data import has_behavioral_metadata, load_dataset


# Observation symbol mapping:
# Symbol 0: Moderate rating (2, 3, 4) + Normal time gap (> 3 days)
# Symbol 1: Extreme rating (1, 5) + Normal time gap (> 3 days)
# Symbol 2: Moderate rating (2, 3, 4) + Burst time gap (<= 3 days)
# Symbol 3: Extreme rating (1, 5) + Burst time gap (<= 3 days) [High Spam Indicator]


def discretize_observation(rating: float, time_gap_days: float) -> int:
    """Discretize rating and time gap into categorical emission symbol (0 to 3)."""
    is_extreme = rating in (1.0, 5.0) or rating <= 1.5 or rating >= 4.5
    is_burst = time_gap_days <= 3.0

    if not is_extreme and not is_burst:
        return 0
    elif is_extreme and not is_burst:
        return 1
    elif not is_extreme and is_burst:
        return 2
    else:
        return 3


class ClassicalHMM:
    """Two-state Hidden Markov Model with Forward, Backward, and Viterbi algorithms."""

    def __init__(
        self,
        n_states: int = 2,
        n_observations: int = 4,
        start_prob: Optional[np.ndarray] = None,
        trans_mat: Optional[np.ndarray] = None,
        emission_mat: Optional[np.ndarray] = None,
    ):
        self.n_states = n_states
        self.n_observations = n_observations

        # State 0: Normal, State 1: Spammer
        self.start_prob = (
            start_prob if start_prob is not None else np.array([0.80, 0.20])
        )
        self.trans_mat = (
            trans_mat
            if trans_mat is not None
            else np.array([[0.88, 0.12], [0.25, 0.75]])
        )
        self.emission_mat = (
            emission_mat
            if emission_mat is not None
            else np.array(
                [
                    [0.55, 0.30, 0.10, 0.05],  # Normal reviewer emissions
                    [0.05, 0.20, 0.25, 0.50],  # Spammer emissions
                ]
            )
        )

    def forward(self, observations: List[int]) -> Tuple[float, np.ndarray]:
        """Forward algorithm with log-domain scaling to prevent numerical underflow."""
        T = len(observations)
        if T == 0:
            return 0.0, np.zeros((0, self.n_states))

        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.start_prob * self.emission_mat[:, observations[0]]
        scale = np.zeros(T)
        scale[0] = np.sum(alpha[0]) or 1e-12
        alpha[0] /= scale[0]

        for t in range(1, T):
            obs = observations[t]
            for j in range(self.n_states):
                alpha[t, j] = (
                    np.sum(alpha[t - 1] * self.trans_mat[:, j])
                    * self.emission_mat[j, obs]
                )
            scale[t] = np.sum(alpha[t]) or 1e-12
            alpha[t] /= scale[t]

        log_likelihood = float(np.sum(np.log(scale)))
        return log_likelihood, alpha

    def backward(self, observations: List[int]) -> np.ndarray:
        """Backward algorithm returning backward variable beta."""
        T = len(observations)
        if T == 0:
            return np.zeros((0, self.n_states))

        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            obs = observations[t + 1]
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.trans_mat[i, :] * self.emission_mat[:, obs] * beta[t + 1, :]
                )
            s = np.sum(beta[t]) or 1e-12
            beta[t] /= s

        return beta

    def viterbi(self, observations: List[int]) -> Tuple[List[int], float]:
        """Viterbi algorithm for decoding the most probable hidden state path."""
        T = len(observations)
        if T == 0:
            return [], 0.0

        # Work in log domain to prevent underflow
        log_pi = np.log(np.maximum(self.start_prob, 1e-12))
        log_A = np.log(np.maximum(self.trans_mat, 1e-12))
        log_B = np.log(np.maximum(self.emission_mat, 1e-12))

        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = log_pi + log_B[:, observations[0]]

        for t in range(1, T):
            obs = observations[t]
            for j in range(self.n_states):
                scores = delta[t - 1] + log_A[:, j]
                best_prev = int(np.argmax(scores))
                delta[t, j] = scores[best_prev] + log_B[j, obs]
                psi[t, j] = best_prev

        # Backtrack
        path = [0] * T
        path[-1] = int(np.argmax(delta[-1]))
        best_log_prob = float(delta[-1, path[-1]])

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, best_log_prob

    def fit_empirical(self, user_sequences: List[Tuple[List[int], int]]):
        """Estimate parameters empirically from labeled user sequences."""
        # user_sequences: list of (observations, label) where label: 0=normal, 1=spammer
        normal_emiss = np.zeros(self.n_observations) + 1.0  # Laplace smoothing
        spam_emiss = np.zeros(self.n_observations) + 1.0

        normal_trans = np.zeros((2, 2)) + 1.0
        spam_trans = np.zeros((2, 2)) + 1.0

        for obs_seq, label in user_sequences:
            if not obs_seq:
                continue
            for o in obs_seq:
                if label == 1:
                    spam_emiss[o] += 1
                else:
                    normal_emiss[o] += 1

            for t in range(len(obs_seq) - 1):
                # Transition proxy
                curr_state = label
                next_state = label
                if label == 1:
                    spam_trans[curr_state, next_state] += 1
                else:
                    normal_trans[curr_state, next_state] += 1

        self.emission_mat[0] = normal_emiss / np.sum(normal_emiss)
        self.emission_mat[1] = spam_emiss / np.sum(spam_emiss)


def extract_user_sequences(df: pd.DataFrame) -> Dict[str, Dict]:
    """Extract review timelines and discrete observation sequences per user."""
    if not has_behavioral_metadata(df):
        raise ValueError("Dataset lacks reviewer metadata columns.")

    data = df.copy()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(3.0)
    data["parsed_date"] = pd.to_datetime(data.get("review_date", "2024-01-01"), errors="coerce")

    # Sort chronological per user
    data = data.sort_values(["user_id", "parsed_date"])

    user_dict = {}
    for user_id, group in data.groupby("user_id"):
        ratings = group["rating"].tolist()
        dates = group["parsed_date"].dt.strftime("%Y-%m-%d").tolist()
        texts = group["text"].tolist()
        labels = group["label"].tolist() if "label" in group.columns else [0] * len(group)

        # Compute time gap between reviews in days
        diffs = group["parsed_date"].diff().dt.total_seconds() / 86400.0
        time_gaps = [round(float(d), 1) if pd.notna(d) else 30.0 for d in diffs]

        # Convert to discrete observations
        observations = [
            discretize_observation(r, g) for r, g in zip(ratings, time_gaps)
        ]

        user_dict[str(user_id)] = {
            "user_id": str(user_id),
            "ratings": ratings,
            "dates": dates,
            "time_gaps": time_gaps,
            "texts": texts,
            "labels": labels,
            "observations": observations,
        }

    return user_dict


def train_and_save_hmm(
    dataset_name: str = "sample",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Train HMM on user sequence dataset and persist parameters."""
    if output_dir is None:
        output_dir = Path.cwd() / "reviewguard" / "models"
        if not output_dir.exists():
            output_dir = Path.cwd() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_dataset(dataset_name)

    if not meta["has_metadata"]:
        return {
            "requires_metadata": True,
            "message": (
                "Sequence Analysis requires a dataset with reviewer metadata (user_id, review_date). "
                "The current dataset does not include reviewer identities. Please switch to the "
                "Synthetic Sample Reviews dataset."
            ),
        }

    user_sequences = extract_user_sequences(df)
    hmm = ClassicalHMM()

    # Train empirical parameters
    seq_tuples = []
    for u_info in user_sequences.values():
        majority_label = 1 if np.mean(u_info["labels"]) >= 0.5 else 0
        seq_tuples.append((u_info["observations"], majority_label))

    hmm.fit_empirical(seq_tuples)

    # Pre-analyze all users
    user_results = []
    for user_id, u_info in user_sequences.items():
        obs = u_info["observations"]
        if not obs:
            continue

        log_lik, alpha = hmm.forward(obs)
        path, viterbi_score = hmm.viterbi(obs)

        spam_state_ratio = float(np.mean(path)) if path else 0.0
        # Posterior spam probability
        spam_prob = round(float(np.mean(alpha[:, 1])) if len(alpha) > 0 else spam_state_ratio, 3)

        user_results.append(
            {
                "user_id": user_id,
                "review_count": len(obs),
                "avg_rating": round(float(np.mean(u_info["ratings"])), 2),
                "avg_time_gap": round(float(np.mean(u_info["time_gaps"])), 1),
                "log_likelihood": round(log_lik, 2),
                "spam_risk_score": spam_prob,
                "viterbi_spam_steps": int(sum(path)),
                "verdict": "Suspicious Sequence" if spam_prob > 0.4 else "Normal Sequence",
            }
        )

    # Sort users by review count and spam score
    user_results.sort(key=lambda u: (u["review_count"], u["spam_risk_score"]), reverse=True)

    output = {
        "requires_metadata": False,
        "n_states": 2,
        "state_names": ["Normal Reviewer", "Spammer / Burst"],
        "transition_matrix": [
            [round(float(v), 3) for v in row] for row in hmm.trans_mat
        ],
        "emission_matrix": [
            [round(float(v), 3) for v in row] for row in hmm.emission_mat
        ],
        "start_probabilities": [round(float(v), 3) for v in hmm.start_prob],
        "total_users": len(user_sequences),
        "user_summaries": user_results[:50],
    }

    with open(output_dir / "hmm_model.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def get_user_hmm_analysis(
    user_id: str,
    dataset_name: str = "sample",
) -> Dict:
    """Compute detailed step-by-step HMM state trajectory and Viterbi path for a specific user."""
    df, meta = load_dataset(dataset_name)

    if not meta["has_metadata"]:
        return {
            "requires_metadata": True,
            "message": "Sequence Analysis requires behavioral metadata dataset.",
        }

    user_sequences = extract_user_sequences(df)
    u_info = user_sequences.get(str(user_id))

    if not u_info:
        # Fallback to first available user
        if user_sequences:
            user_id = next(iter(user_sequences))
            u_info = user_sequences[user_id]
        else:
            return {"error": f"User {user_id} not found."}

    hmm = ClassicalHMM()
    obs = u_info["observations"]
    log_lik, alpha = hmm.forward(obs)
    beta = hmm.backward(obs)
    path, viterbi_score = hmm.viterbi(obs)

    # Build timeline steps
    timeline = []
    symbol_desc = {
        0: "Moderate rating + Normal gap",
        1: "Extreme rating (1 or 5★) + Normal gap",
        2: "Moderate rating + Burst gap (≤3 days)",
        3: "Extreme rating (1 or 5★) + Burst gap (≤3 days)",
    }

    for t in range(len(obs)):
        post_spam = float(alpha[t, 1] * beta[t, 1]) / (
            float(np.sum(alpha[t] * beta[t])) or 1e-12
        )
        timeline.append(
            {
                "step": t + 1,
                "date": u_info["dates"][t],
                "rating": u_info["ratings"][t],
                "time_gap_days": u_info["time_gaps"][t],
                "observation_code": obs[t],
                "observation_desc": symbol_desc.get(obs[t], "Unknown"),
                "viterbi_state": int(path[t]),
                "viterbi_state_name": "Spammer State"
                if path[t] == 1
                else "Normal State",
                "posterior_spam_prob": round(float(post_spam), 3),
                "review_snippet": u_info["texts"][t][:80] + "..."
                if len(u_info["texts"][t]) > 80
                else u_info["texts"][t],
            }
        )

    spam_prob = round(float(np.mean(path)), 3)

    return {
        "requires_metadata": False,
        "user_id": str(user_id),
        "total_reviews": len(obs),
        "overall_log_likelihood": round(log_lik, 3),
        "spam_risk_score": spam_prob,
        "classification": "Flagged as Spammer"
        if spam_prob > 0.4
        else "Normal Organic Reviewer",
        "transition_matrix": [
            [round(float(v), 3) for v in row] for row in hmm.trans_mat
        ],
        "timeline": timeline,
    }
