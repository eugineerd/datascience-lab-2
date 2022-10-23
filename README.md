### Выбор метрики
В качестве основной метрики был выбран $R^2$ score, т.к. это удобная абсолютная метрика.
### Результаты для тестовой выборки размером 20% от тренировочной
|Path                                   |MAPE     |r2 score|
|--|--|--|
|metrics/test/catboost_tr.json          |0.08711  |0.93079|
|metrics/test/catboost_native.json      |0.09291  |0.92248|
|metrics/test/extra_trees.json          |0.09742  |0.91269|