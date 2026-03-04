## About FEV Explorer

FEV Explorer is an interactive viewer for the
[FEV Benchmark](https://github.com/autogluon/fev), created by [Magnus Ross](https://magnusross.github.io). In benchmark papers, the focus is usually on aggregated error statistics, but it is also important to understand what the forecasts of different models look like.  In the app, you can browse benchmark tasks, inspect individual
forecast windows, and compare model predictions against ground-truth data.


My main concern when building this app was to understand how the forecasts of the best foundation models compare to statistical baselines. Given that, some notes on the included models:
- The only foundation models currently included are from the 
[Chronos](https://github.com/amazon-science/chronos-forecasting) family: `chronos-2` and `chronos-bolt-small`
- AutoARIMA is not included because it takes way too long to run.
- I have included AutoAR from ["Specialized Foundation Models Struggle to Beat Supervised Baselines"](https://arxiv.org/abs/2411.02796) as an alternative linear model, which was shown to perform better than AutoARIMA 


If you want more models please drop me an email, and I can add them.