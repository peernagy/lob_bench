# lob_bench
Benchmarking library for generative models of Limit Order Book data (LOBSTER)

The benchmarking library consists of two parts.
I. The evaluation of distributions of 1-Dimensional scores taken from real and generated sequences of data. 
    a. A measure of dissimilarity between data sets is obtained through the distance between the two distributions: implemented distance metrics are L1 and Wasserstein.
    b. Currently implemented score functions are:
        i. Spread - Difference between the best ask and best bid price.
        ii. Interarrival time - Time between two successeive orders, of any type, in ms. 
        iii. Orderbook imbalance - Difference in volume available between the best bid and ask prices. 
        iv. Time to cancel - Time between submission of an order and the first modification/cancellation.
        v./vi. Volume - Volume on the ask/bid side for the first n levels (Typically 10).
        vii./viii. Limit order ask/bid depths - Distance of new limit orders from the midprice.
        ix./x. Limit order ask/bid levels - Level of new limit orders (1 to n, n typically 10).
        xi./xii. Cancel order ask/bid depths - Distance of cancel orders from the midprice.
        xiii./xiv. Cancel order ask/bid levels - Level of cancel orders (1 to n, n typically 10).
    c. Conditional score functions are those which evaluate a distribution given that the value of a conditional score are in some interval. So far, the following are implemented:
        i. Ask volume (of first level) conditional on the spread.
        ii. Spread conditional on the time of day (hourly).
        iii. Spread conditional on the volatility (Stdev of returns over a given time interval)
II. Consideration of impact response functions 
    a. The response function is calculated for 6 different event types:
        i. Market orders which do not change the mid-price: MO_0
        ii. Market orders which do change the mid-price: MO_1
        iii. Limit orders which do not change the mid-price: LO_0
        iv. Limit orders which do change the mid-price: LO_1
        v. Cancel orders which do not change the mid-price: CA_0
        vi. Cancel orders which do change the mid-price: CA_1
    b. Given real and generated data-sequences, the response curves are calculated seperately for each data-set and give a qualititative comparison of how the impact of the event types.

## Benchmarking Quickstart

### Data:
The input data for the benchmark can be easily loaded by the data loading object: data_loading.Simple_Loader. 
At initialisation, this object requires three arguments, all of which are file paths:
    i. the location of the data sequences used for conditioning the model at inference, this is optional
    ii. the location of the generated data sequences
    iii. the location of the sequences to which the generated sequences are to be compared. 

Each of these directories must contain an equal number of comma-separated value files containing the messages and orderbooks for each sequence. These files should have a format identical to that of LOBSTER (https://lobsterdata.com/info/DataStructure.php) data, though this is not a prerequisite for the model to be evaluated. 

### Scores
Once the loader is configured with the filepaths to data following the above requirements, it becomes very east to use the benchmark. 

To evaluate the distances between distributions, the score functions that may generate 1-D distributions must be defined. This is done by using a nested Dict of the following form:

{"ScoreName_1" : {"fn" : Callable score_func, "Discrete" : Optional Bool discrete_output }, 
 "ScoreName_2" : ...
 
 "Cond_ScoreName_1: {"eval" : {"fn" : Callable score_func, "Discrete" : Optional Bool discrete_output },
                     "cond" : {"fn" : Callable score_func, "Discrete" : Optional Bool discrete_output }},
 "Cond_ScoreName_2: ...
 }

 Whereby the conditional score functions require two callable functions to evaluate both the metric and that on which it is conditioned. The pre-implemented score functions can be found in the eval.py file. Any custom function may also be used.

 The other required definition dictionary is the distance metrics. Two distances are implemented in the metrics.py file and can be used by defining the metrics dictionary in the form:

{"MetricName_1": Callable metric_function,
  "MetricName_2: ...
}

 Once these two dictionaries are defined with either the implemented functions, or any custom score/metric function, the benchmarks can be run using the run_benchmark function in the scoring.py file. 

The function returns a tuple containing 
    A. The distances for each metric and score as well as bootstrapped confidence intervals
    B. The raw score values as well as there bin index for histograms for each entry in the real and generated sequences for each score. 
    C. Plotting functions for each score to plot the histograms thereof.

Examples of these outputs are given in the tutorial notebook. 

### Impact functions
The impact resonse functions are calculated by simply providing the data loader object to the impact_compare function in the impact.py file.

This filters and labels the data to only consider the events of note, details of which are in the paper. The response functions are then plotted with the 99% confidence interval depicted by the shaded regions. 


### Discriminator Scoring


