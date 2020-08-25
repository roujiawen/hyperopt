import ray
import pandas as pd


@ray.remote
def distribute(func, *args, **kwargs):
    """
    A wrapper to use remote workers to compute func and return the results.
    """
    output = func(*args, **kwargs)
    return output

class Experiment:
    """
    A hyperparameter search experiment for a machine learning model that can
    be run locally or in a distributed environment.

    Methods:
        search: perform hyperparameter search.
        summary: visualize the results.

    """
    def __init__(self, trainer, config, output_path=None):
        """
        Parameters:
            trainer (Callable): a function that contains scripts for model
                building and training. This function must takes in a set of
                hyperparameters and returns an optimization objective metric.
            config (dict): contains user specifications of search strategy;
                for example, which hyperparameters to optimize and the size
                and resolution of the search space.
            output_path (str): where to save logs and summary files from the
                experiment.
        """
        self.trainer = trainer
        self.config = config
        self.output_path = output_path if output_path else "results/"
        # summary table will be a pd.DataFrame, each row containing a set of
        # hyperparameter values and the test accuracy from corresponding model
        self.summary_table = None
        # create directory for logs
        self._create_output_dir()
        # choose search strategy
        if self.config["search_strategy"] == "grid":
            self.search = self._grid_search
        else:
            # Future feature: connect with more sophisticated optimizers
            self.search = self._interface_optimizers
        # choose local-single/local-multi/distributed mode
        if self.config["computing_mode"] == "local-single":
            # use only a single thread
            ray.init(local_mode=True)
        elif self.config["computing_mode"] == "local-multi":
            # parallelize locally
            ray.init()
        else: # "distributed"
            ray.init(address="auto")

    def _create_output_dir(self):
        pass

    def _generate_hpsets(self):
        """
        A generator function to convert user-specified grid_search_space into
        all possible hyperparameter sets, which are generated one at a time.

        For example, given

            config["grid_search_space"] = {
                "param1": [1, 10, 20, ...100],
                "param2": [True, False],
                "param3": [0.1, 0.3, 1, ...1000]
            }

        list(Experiment._generate_hpsets()) will generate
            [
                {"param1": 1, "param2": True, "param3": 0.1},
                {"param1": 1, "param2": True, "param3": 0.3},
                ...
                {"param1": 100, "param2": False, "param3": 1000}
            ]

        """
        from itertools import product
        search_space = self.config["grid_search_space"]
        # generate all combinations of search values
        all_combinations = product(*search_space.values())
        # grab names of all hyperparameters
        names = search_space.keys()
        # yielding one hyperparameter set at a time, in dictionary format
        for values in all_combinations:
            hpset = dict(zip(names, values))
            yield hpset


    def _grid_search(self):
        # Initialize empty buffer for rows of the summary table
        self._table_rows = []

        # Generate all possible hyperparameter combinations and distribute them
        # for remote workers to train
        working_workers = []
        for hpset in self._generate_hpsets():
            worker = distribute.remote(self.trainer, hpset)
            working_workers.append(worker)

        # Waiting to get results back. Save results to disk in batches of size n.
        every_n = self.config["grid_search_settings"]["save_every_n_outputs"]
        while working_workers:
            done_workers, working_workers = ray.wait(working_workers,
                                                     num_returns=every_n)
            results = ray.get(done_workers)
            # Save detailed logs to disk for each record
            self._save(results)
            # Add a new row in summary table: hyperparameter values + accuracy
            self._add_to_summary(results)

        # Generate summary table from row buffers
        self._make_summary_table()
        # Save summary table to disk
        self._save_summary_table()

    def _save(self, results):
        """
        Save individual records to disk.
        """
        pass

    def _add_to_summary(self, results):
        """
        Parameters:
            results: List[(hpset: Mapping[str, Any], accuracy: float, logs: Any)]

        Warning: hpset is modified after this function is executed
        """
        for hpset, accuracy, _ in results:
            hpset["accuracy"] = accuracy

        self._table_rows += [row for row, _, _ in results]

    def _save_summary_table(self):
        pass

    def _make_summary_table(self):
        """
        Turn buffered rows into an actual pd.DataFrame table.
        """
        self.summary_table = pd.DataFrame(self._table_rows)

    def _interface_optimizers(self):
        """
        Connect with existing open-source search algorithms.
        """
        pass

    def summary(self):
        """
        Print top results from the summary table.
        Plot results.
        """
        # print table, showing the top results
        table = self.summary_table
        print(self.summary_table.nlargest(20, 'accuracy'))
        print("...")
        print("[{} rows x {} columns]".format(table.shape[0], table.shape[1]))
        # plot results
        # [TBC]
