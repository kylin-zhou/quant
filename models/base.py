# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from typing import Text, Union


class BaseModel():
    """Modeling things"""

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """Make predictions after modeling things"""

    def __call__(self, *args, **kwargs) -> object:
        """leverage Python syntactic sugar to make the models' behaviors like functions"""
        return self.predict(*args, **kwargs)


class Model(BaseModel):
    """Learnable Models"""

    def fit(self):
        """
        Learn model from the base model

        .. note::

            The attribute names of learned model should `not` start with '_'. So that the model could be
            dumped to disk.

        The following code example shows how to retrieve `x_train`, `y_train` and `w_train` from the `dataset`:

            .. code-block:: Python

                # get features and labels
                df_train, df_valid = dataset.prepare(
                    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                x_train, y_train = df_train["feature"], df_train["label"]
                x_valid, y_valid = df_valid["feature"], df_valid["label"]

                # get weights
                try:
                    wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"],
                                                           data_key=DataHandlerLP.DK_L)
                    w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
                except KeyError as e:
                    w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
                    w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed data from model training.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self) -> object:
        """give prediction given Dataset

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed dataset from model training.

        segment : Text or slice
            dataset will use this segment to prepare data. (default=test)

        Returns
        -------
        Prediction results with certain type such as `pandas.Series`.
        """
        raise NotImplementedError()