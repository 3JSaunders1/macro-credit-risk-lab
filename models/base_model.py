from abc import ABC, abstractmethod


class BaseMacroModel(ABC):

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def forecast(self, steps=1):
        pass

    @abstractmethod
    def irf(self):
        """
        MUST return:
        {
            "irfs": np.ndarray,
            "fevd": np.ndarray | None,
            "type": str
        }
        """
        pass