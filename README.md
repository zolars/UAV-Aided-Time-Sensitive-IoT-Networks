# Model for Priority-based Trajectory Planning for UAV-sensorTime-sensitive Network

## Description

Some path schedule solutions specifically for UAV-sensorTime-sensitive Network. The code is used to evaluate the models' performance and time complexity.

## Parameters
 
* length_range: The range of map's length and width.
* priority_range: The range of possible sensors' priority.
* s: ![]([http://latex.codecogs.com/gif.latex?\\\sqrt{{(r + R)}^2 - h^2}](https://latex.codecogs.com/gif.latex?\sqrt{{(r&space;&plus;&space;R)}^2&space;-&space;h^2}))
* v: The speed of UAV.
* period: One period named `t` in the paper.
* t_limit: The working-time limitation which is caused by fuel.
* max_time: The total time the system run.
* seed: Random number seed for generate random sensors lists.

## Usage
1. Install Python and dependencies with [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    ```
    $ conda -V
    conda 4.7.10
    ```

2. Install Python and dependencies with conda.
    ```
    $ conda env create -f environment.yml
    $ conda activate UAV
    (UAV) $ python -V
    Python 3.7.3
    ```

    or
    ```
    $ conda env update -f environment.yml
    ```

3. Run Python script. The results save at `./out/`
    ```
    (UAV) $ python model.py
    ```

4. If you wanna remove conda environment:
    ```
    (UAV) $ conda deactivate
    $ conda remove -n UAV --all
    ```