import numpy as np
import pandas as pd


def get_resampled_data(data: pd.Series, n_resamples: int, random_state: int = None) -> pd.DataFrame:
    """
    Получить бутстрапированную таблицу через пуассоновское распределение
    :param n_resamples: количество бутстрапов
    :param data: исходная статистика
    :return:
    """
    np.random.seed(random_state)
    limit_value = 1  # https://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
    data = data.to_frame("data")
    data["resample_count_by_id"] = [
        [(i, count) for i, count in enumerate(np.random.poisson(limit_value, n_resamples))]
        for _ in range(data.shape[0])
    ]
    data = data.explode("resample_count_by_id", ignore_index=True)
    data["resample_id"] = data["resample_count_by_id"].apply(lambda x: x[0])
    data["count_by_resample_id"] = data["resample_count_by_id"].apply(lambda x: x[1])

    return data.drop(columns=["resample_count_by_id"])
