Weighted F1-Score
=================

We use weighted F1-score because dataset classes are imbalanced.

Why Weighted F1?
----------------
- Gives importance based on class frequency
- Avoids bias toward majority class
- Better performance indicator

Formula:
Weighted F1 = Σ (support_i / total_samples) * F1_i
