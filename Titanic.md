Titanic: Machine Learning
=========================

> 第一个machine learning 的toy，根据现有的一部分数据，存活情况，年龄，性别等特征，预测未知部分乘客是否死亡。

这个项目是kaggle平台上的一个[beginning的项目](https://www.kaggle.com/c/titanic)，作为熟悉kaggle以及machine learning的一个入门，接下来从最近认识的角度来讲一下整个项目的思路。

> 数据分析很重要

数据组成在官网上有详细的解释，总之数据的总体特征如下：

 - 部分数据缺失，需要找办法预测，补回
 - 部分特征对最后预测乘客是否死亡起不到作用，例如乘客的ID，需要剔除
 - 数据特征不是很好，需要我们根据数据创造一些新的特征，以更好的满足样本的需求

针对这些问题，我们可以通过画图等方式直观展现某些特征对数据的影响。

    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();
    

> 模型，准确度

这个toy在判断一些乘客的存活情况，属于分类，回归问题。我们使用的模型有SVM，KNN，Decision tree，random forest，logistics regression等。最后random forest的效果最好，在测试集上的准确率为：**86.76**

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    acc_random_forest

**收获**
------

跟着[教程](https://www.kaggle.com/startupsci/titanic-data-science-solutions)走了一遍程序，最大的体会是，**特征分析上**，要把一些不需要的特征删除,一些**特征的整合**，以及最后特征的**序数化**。第二个是要分析清楚问题的特征，选用合适的机器学习算法。除此之外也算躺了一遍水，大概懂得了过程。

不足
--

 - 数据处理使用panda模块，需要学习。
 - 对特征的处理上，自己毫无头绪，需要加深对问题的了解
 - 在数据对比上，对matplotlib，seaborn的模块也需要学习一下，不会使用它们
 - 对问题的类型有点含糊，没能选出可用的模型。
 - 在sklearn这个库，使用也很陌生，不会使用

下一步工作
-----

再找一个项目来做做，找找感觉。