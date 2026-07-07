import re

file_path = '/home/saurav/Documents/Research/apex_journal/papers_manuscript/paper.md'
with open(file_path, 'r') as f:
    text = f.read()

# 1. Ref 17 (LightGBM)
text = text.replace(
    "17. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. [https://scholar.google.com/scholar?q=LightGBM:+A+highly+efficient+gradient+boosting+decision+tree](https://scholar.google.com/scholar?q=LightGBM:+A+highly+efficient+gradient+boosting+decision+tree)",
    "17. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30. <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf>"
)

# 2. Ref 20 URL formatting
text = text.replace(
    "20. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf",
    "20. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. <https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf>"
)

# 3. Discrepancy explanation in Abstract
abs_old = "In the normalized target growth-rate space upon which it was trained, the SVR achieved a Mean Absolute Error (MAE) of 15.8% versus the traditional baseline's 20.8%. When reconstructed into absolute currency, this translates to an average error of Rs 91,442 for SVR compared to Rs 96,114 for the baseline, demonstrating modest but robust statistical significance"
abs_new = "In the normalized target growth-rate space upon which it was trained, the SVR achieved a Mean Absolute Error (MAE) of 15.8% versus the traditional baseline's 20.8% (a ~24% relative improvement). When reconstructed into absolute currency, this translates to an average error of Rs 91,442 for SVR compared to Rs 96,114 for the baseline (a ~5% relative improvement); this compression in relative impact occurs because absolute rupee errors are naturally dominated by the immense scale of a few massive-capacity firms. These results demonstrated modest but robust statistical significance"
text = text.replace(abs_old, abs_new)

# 3b. Discrepancy explanation in Section 5.1
body_old = "In the original normalized growth-rate space, the SVR achieved an MAE of 15.8% compared to the baseline's 20.8%. Reconstructed into rupees, the SVR achieved an out-of-sample Mean Absolute Error (MAE) of **Rs 91,442**, outperforming the static assumption baseline (**Rs 96,114**)."
body_new = "In the original normalized growth-rate space, the SVR achieved an MAE of 15.8% compared to the baseline's 20.8%, representing a relative improvement of approximately 24%. Reconstructed into rupees, the SVR achieved an out-of-sample Mean Absolute Error (MAE) of **Rs 91,442**, outperforming the static assumption baseline (**Rs 96,114**) by a relative margin of about 5%. This discrepancy between the ~24% improvement in growth-rate space and the ~5% improvement in absolute currency space occurs because a few large-capacity firms inherently generate outsized absolute rupee errors. Consequently, even a highly accurate percentage growth prediction for a massive firm will carry a large rupee penalty, diluting the perceived relative improvement when aggregated at the scale of absolute currency."
text = text.replace(body_old, body_new)


with open(file_path, 'w') as f:
    f.write(text)
