from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class HyperOptTuning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass


class TitanicDataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "Cabin" in X.columns:
            X.drop('Cabin', axis=1, inplace=True)
        if "Ticket" in X.columns:
            X.drop('Ticket', axis=1, inplace=True)
        if "PassengerId" in X.columns:
            X.drop('PassengerId', axis=1, inplace=True)

        age_medians = X.groupby(['Pclass', 'Sex', 'Parch', 'SibSp'])['Age'].mean()

        def impute_age(row):
            if pd.isnull(row['Age']):
                __x__ = age_medians.loc[row['Pclass'], row['Sex'], row['Parch'], row['SibSp']]
                if pd.isnull(__x__):
                    __x__ = X.groupby(['Pclass', 'Sex'])['Age'].mean().loc[row['Pclass'], row['Sex']]
                return __x__
            else:
                return row['Age']

        def filter_name(row):
            if "Mr." in row['Name']:
                return "Mr"
            if "Ms." in row['Name']:
                return "Ms"
            elif "Master." in row['Name']:
                return "Master"
            elif "Mrs." in row['Name']:
                return "Mrs"
            elif "Miss." in row['Name']:
                return "Miss"
            elif "Don." in row['Name']:
                return "Miss"
            elif "Rev." in row['Name']:
                return "Rev"
            elif "Dr." in row['Name']:
                return "Dr"
            elif "Major." in row['Name']:
                return "Major"
            elif "Lady." in row['Name']:
                return "Lady"
            elif "Sir." in row['Name']:
                return "Sir"
            elif "Col." in row['Name']:
                return "Col"
            elif "Capt." in row['Name']:
                return "Capt"
            elif "Mlle." in row['Name']:
                return "Mlle"
            else:
                return "Mr"

        X['Age_Imputed'] = X.apply(impute_age, axis=1)
        X['Name'] = X.apply(filter_name, axis=1)

        X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

        return X


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col])

        return X

    def get_label_encoder(self, col=None):
        if col is None:
            return self.encoders
        else:
            return self.encoders[col]
