import ray
from hyperopt import distribute

class testDistribute:
    def testCorrectness():
        futures = [distribute.remote(sorted, [3,1,2], reverse=True) for i in range(100)]
        assert ray.get(futures) == [[3,2,1]] * 100
    def testTiming():
        """
        Assign a function that sleeps for 10 seconds to N workers, where N is
        the number of threads available in the environment. Check that it takes
        much less than 10*N seconds to complete all the tasks -> this shows that
        parallelism is working.
        """
        pass

class testHyperopt:
    """
    Not enough time to implement this but here are a few ideas to test
    the optimizatin toolkit:
      - use simple functions with known optimum as trainer functions.
        For example, we could take function f(x,y) = -(x^2 + y^2) and pretend
        that x and y are `hyperparameters` and f is `accuracy`, and feed this
        function into the optimizer.
      - use simple statistical models to generate fake data given chosen hyper-
        parameters and then use the optimizer to recover the hyperparameters.
        For example, (I tried this earlier but didn't get it to work) generate
        fake time series data using ARIMA with specified order (p,d,q), and then
        try to recover the order using hyperopt.
    """
    pass
