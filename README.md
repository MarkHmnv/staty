# Staty
The Staty library provides functions to calculate statistical measures on a dataset. It runs on Python 3.
**This library is a personal pet project created for the purpose of learning Data Science and Statistics concepts.**

## Usage
Here is an example:
```python
import staty

data = [2, 4, 6, 8]
print(staty.stderr(data)) # 1.2909944487358056
```
## Functions

`Staty` provides the following functions:

1. **`mean(data: List[Union[int, float]]) -> float`**
   - Calculates the mean of a list of numbers. 

2. **`var(data: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the variance of a given dataset.

3. **`stdev(data: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the standard deviation of a given list of numbers.

4. **`stderr(data: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the standard error of a data set.

5. **`median(data: List[Union[int, float, str]]) -> Union[float, Tuple[float, float]]`**
   - Calculates the median value(s) of the given list of data.

6. **`mode(data: List[Union[int, float, str]]) -> Union[float, str, List]`**
   - Finds the mode(s) of a given list of data.

7. **`cv(data: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the coefficient of variation for a given list of data.

8. **`cov(data_x: List[Union[int, float]], data_y: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the covariance between two sets of data points.

9. **`correlation_r(data_x: List[Union[int, float]], data_y: List[Union[int, float]], is_sample: bool = True) -> float`**
   - Calculates the Pearson correlation coefficient between two sets of data.

10. **`zscore(data: List[Union[int, float]], is_sample: bool = True) -> List[float]`**
    - Calculate the z-scores of a list of data points.

