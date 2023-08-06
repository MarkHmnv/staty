from collections import Counter
from typing import Union, List, Tuple
from math import pow, sqrt


def mean(data: List[Union[int, float]]) -> float:
    """
    Calculate the mean of a list of numbers.

    :param data: A list of numbers.
    :return: The mean value as a float.

    :raises ValueError: If the length of the input data is less than 2.

    """
    n = len(data)
    _validate_min_len(n)
    return sum(data) / n


def var(data: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculate the variance of a given dataset.

    :param data: A list of numeric values.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The variance of the dataset.

    :raises ValueError: If the length of the input data is less than 2.

    """
    n = len(data)
    _validate_min_len(n)
    m = mean(data)
    return _squared_difference(data, m) / (n - 1 if is_sample else n)


def stdev(data: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculate the standard deviation of a given list of numbers.

    :param data: A list of integers or floats representing the dataset.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The standard deviation as a float value.

    :raises ValueError: If the length of the input data is less than 2.

    """
    _validate_min_len(len(data))
    variance = var(data, is_sample)
    return sqrt(variance)


def stderr(data: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculate the standard error of a data set.

    :param data: A list of integers or floats representing the data set.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The standard error of the data set as a float.

    :raises ValueError: If the length of the input data is less than 2.

    """
    n = len(data)
    _validate_min_len(n)
    std_dev = stdev(data, is_sample)
    return std_dev / sqrt(n)


def median(data: List[Union[int, float, str]]) -> Union[float, Tuple[float, float]]:
    """
    Calculates the median value(s) of the given list of data.

    :param data: A list of elements. Can contain integers, floats, or strings.
    :return: If the number of elements is odd, returns the middle value.
             If the number of elements is even, returns the average of the two middle values.
             If the elements are strings, returns a tuple of the two middle values.

    :raises ValueError: If the length of the input data is less than 2.

    """
    n = len(data)
    _validate_min_len(n)
    data = sorted(data)

    if n % 2 == 0:
        middle_right = n // 2
        middle_left = middle_right - 1
        return (data[middle_left] + data[middle_right]) / 2 if isinstance(data[0], (float, int)) \
            else (data[middle_left], data[middle_right])
    else:
        return data[n // 2]


def mode(data: List[Union[int, float, str]]) -> Union[float, str, List]:
    """
    Finds the mode(s) of a given list of data.

    :param data: A list containing integers, floats, and/or strings.
    :return: The mode(s) of the data.

    :raises ValueError: If the length of the input data is less than 2.

    """
    _validate_min_len(len(data))
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [num for num, count in counts.items() if count == max_count]

    if len(modes) == 1:
        return modes[0]

    return modes


def cv(data: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculate the coefficient of variation for a given list of data.

    :param data: A list of numerical values.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The coefficient of variation as a float value.

    :raises ValueError: If the length of the input data is less than 2.

    """
    _validate_min_len(len(data))
    std_dev = stdev(data, is_sample)
    m = mean(data)
    return std_dev / m


def cov(data_x: List[Union[int, float]], data_y: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculates the covariance between two sets of data points given by `data_x` and `data_y`.

    :param data_x: A list of integers or floats representing the x coordinate values.
    :param data_y: A list of integers or floats representing the y coordinate values.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The covariance between `data_x` and `data_y` as a float.

    :raises ValueError: If the lengths of `data_x` and `data_y` are not equal or the input data is less than 2.

    """
    n = len(data_x)
    _ensure_equal_len(len(data_y), n)
    mx = mean(data_x)
    my = mean(data_y)
    return sum((data_x[i] - mx) * (data_y[i] - my) for i in range(n)) / (n - 1 if is_sample else n)


def correlation_r(data_x: List[Union[int, float]], data_y: List[Union[int, float]], is_sample: bool = True) -> float:
    """
    Calculate the Pearson correlation coefficient between two sets of data.

    :param data_x: A list of numerical values representing one set of data.
    :param data_y: A list of numerical values representing the other set of data.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: The Pearson correlation coefficient between the two sets of data.

    :raises ValueError: If the lengths of `data_x` and `data_y` are not equal or the input data is less than 2.

    """
    n = len(data_x)
    _ensure_equal_len(len(data_y), n)
    covariance = cov(data_x, data_y, is_sample)
    return covariance / (stdev(data_x, is_sample) * stdev(data_y, is_sample))


def zscore(data: List[Union[int, float]], is_sample: bool = True) -> List[float]:
    """
    Calculate the z-scores of a list of data points.

    :param data: A list of numeric data points.
    :param is_sample: Optional. A boolean value indicating whether the data is a sample or population.
    :return: A list of z-scores corresponding to the input data.

    :raises ValueError: If the length of the input data is less than 2.

    """
    _validate_min_len(len(data))
    m = mean(data)
    std_dev = stdev(data, is_sample)
    return [(value - m) / std_dev for value in data]


def _squared_difference(data: List[Union[int, float]], mean_value: float) -> float:
    return sum(pow(value - mean_value, 2) for value in data)


def _ensure_equal_len(n1: int, n2: int):
    _validate_min_len(n1)
    _validate_min_len(n2)
    if n1 != n2:
        raise ValueError("The lists must be of the same length")


def _validate_min_len(n: int):
    if n < 2:
        raise ValueError("The list must contain at least 2 elements")
