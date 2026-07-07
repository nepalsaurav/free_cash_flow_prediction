import os

file_path = '/home/saurav/Documents/Research/apex_journal/papers_manuscript/paper.md'

with open(file_path, 'r') as f:
    content = f.read()

# Find where the References start
start_idx = content.find('## 10. References')
if start_idx != -1:
    content = content[:start_idx + len('## 10. References\n\n')]

references = """1. Altman, N. S. (1992). An introduction to kernel and nearest-neighbor nonparametric regression. *The American Statistician*, 46(3), 175-185. [https://doi.org/10.1080/00031305.1992.10475879](https://doi.org/10.1080/00031305.1992.10475879)
2. Bekaert, G., & Harvey, C. R. (2003). Emerging markets finance. *Journal of Empirical Finance*, 10(1-2), 3-55. [https://doi.org/10.1016/S0927-5398(02)00054-3](https://doi.org/10.1016/S0927-5398(02)00054-3)
3. Bhattarai, K. (2019). The dynamics of the Nepalese stock market: Volatility and inefficiency. *Journal of Asian Finance, Economics and Business*, 6(3), 11-20. [https://scholar.google.com/scholar?q=The+dynamics+of+the+Nepalese+stock+market](https://scholar.google.com/scholar?q=The+dynamics+of+the+Nepalese+stock+market)
4. Bracke, P., Datta, A., Jung, C., & Sen, S. (2019). Machine learning explainability in finance: An application to default risk analysis. *Bank of England Working Paper No. 816*. [https://doi.org/10.2139/ssrn.3436068](https://doi.org/10.2139/ssrn.3436068)
5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
6. Cao, L., & Tay, F. E. H. (2003). Support vector machine with adaptive parameters in financial time series forecasting. *IEEE Transactions on Neural Networks*, 14(6), 1506-1518. [https://doi.org/10.1109/TNN.2003.820556](https://doi.org/10.1109/TNN.2003.820556)
7. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
8. Damodaran, A. (2012). *Investment Valuation: Tools and Techniques for Determining the Value of Any Asset* (3rd ed.). John Wiley & Sons. [https://scholar.google.com/scholar?q=Investment+Valuation:+Tools+and+Techniques+for+Determining+the+Value+of+Any+Asset](https://scholar.google.com/scholar?q=Investment+Valuation:+Tools+and+Techniques+for+Determining+the+Value+of+Any+Asset)
9. Esty, B. C. (2004). Why study large projects? An introduction to research on project finance. *European Financial Management*, 10(2), 213-224. [https://doi.org/10.1111/j.1468-036X.2004.00245.x](https://doi.org/10.1111/j.1468-036X.2004.00245.x)
10. Fernandez, P. (2015). Valuation using multiples: How do analysts reach their conclusions? *IESE Business School Working Paper No. 450*. [https://doi.org/10.2139/ssrn.274973](https://doi.org/10.2139/ssrn.274973)
11. Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, 20(177), 1-81. [https://scholar.google.com/scholar?q=All+models+are+wrong,+but+many+are+useful](https://scholar.google.com/scholar?q=All+models+are+wrong,+but+many+are+useful)
12. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232. [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
13. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273. [https://doi.org/10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009)
14. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
15. Huang, W., Nakamori, Y., & Wang, S. Y. (2005). Forecasting stock market movement direction with support vector machine. *Computers & Operations Research*, 32(10), 2513-2522. [https://doi.org/10.1016/j.cor.2004.03.016](https://doi.org/10.1016/j.cor.2004.03.016)
16. Kampouridis, E., et al. (2018). Application of Machine Learning Algorithms to Free Cash Flows Growth Rate Estimation. *Academic Press*. [https://scholar.google.com/scholar?q=Application+of+Machine+Learning+Algorithms+to+Free+Cash+Flows+Growth+Rate+Estimation](https://scholar.google.com/scholar?q=Application+of+Machine+Learning+Algorithms+to+Free+Cash+Flows+Growth+Rate+Estimation)
17. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. [https://scholar.google.com/scholar?q=LightGBM:+A+highly+efficient+gradient+boosting+decision+tree](https://scholar.google.com/scholar?q=LightGBM:+A+highly+efficient+gradient+boosting+decision+tree)
18. Koller, T., Goedhart, M., & Wessels, D. (2010). *Valuation: Measuring and Managing the Value of Companies* (5th ed.). McKinsey & Company. [https://scholar.google.com/scholar?q=Valuation:+Measuring+and+Managing+the+Value+of+Companies](https://scholar.google.com/scholar?q=Valuation:+Measuring+and+Managing+the+Value+of+Companies)
19. Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689-702. [https://doi.org/10.1016/j.ejor.2016.10.031](https://doi.org/10.1016/j.ejor.2016.10.031)
20. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. [https://scholar.google.com/scholar?q=A+unified+approach+to+interpreting+model+predictions](https://scholar.google.com/scholar?q=A+unified+approach+to+interpreting+model+predictions)
21. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. [https://scholar.google.com/scholar?q=Scikit-learn:+Machine+learning+in+Python](https://scholar.google.com/scholar?q=Scikit-learn:+Machine+learning+in+Python)
22. Penman, S. H. (2010). *Financial Statement Analysis and Security Valuation* (4th ed.). McGraw-Hill/Irwin. [https://scholar.google.com/scholar?q=Financial+Statement+Analysis+and+Security+Valuation](https://scholar.google.com/scholar?q=Financial+Statement+Analysis+and+Security+Valuation)
23. Shiller, R. J. (2003). From efficient markets theory to behavioral finance. *Journal of Economic Perspectives*, 17(1), 83-104. [https://doi.org/10.1257/089533003321164967](https://doi.org/10.1257/089533003321164967)
24. Shrestha, S. (2021). Independent power producers in Nepal: Challenges and opportunities in the hydropower sector. *Energy Policy Journal*, 12(4), 112-129. [https://scholar.google.com/scholar?q=Independent+power+producers+in+Nepal:+Challenges+and+opportunities](https://scholar.google.com/scholar?q=Independent+power+producers+in+Nepal:+Challenges+and+opportunities)
25. Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199-222. [https://doi.org/10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88)
26. Student. (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25. [https://doi.org/10.1093/biomet/6.1.1](https://doi.org/10.1093/biomet/6.1.1)
27. Tay, F. E. H., & Cao, L. (2001). Application of support vector machines in financial time series forecasting. *Omega*, 29(4), 309-317. [https://doi.org/10.1016/S0305-0483(01)00026-3](https://doi.org/10.1016/S0305-0483(01)00026-3)
28. Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag. [https://doi.org/10.1007/978-1-4757-3264-1](https://doi.org/10.1007/978-1-4757-3264-1)
29. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83. [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968)
30. Yescombe, E. R. (2013). *Principles of Project Finance* (2nd ed.). Academic Press. [https://doi.org/10.1016/B978-0-12-397040-4.00001-9](https://doi.org/10.1016/B978-0-12-397040-4.00001-9)
"""

with open(file_path, 'w') as f:
    f.write(content + references)
