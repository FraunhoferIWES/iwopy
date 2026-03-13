from abc import ABCMeta, abstractmethod
from pathlib import Path

from .base import Base


class Pipeline(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization pipelines.

    An optimization pipeline is a collection of optimization problems
    and optimmizers that are being run one after another. Each step
    of this process is called a stage.

    Attributes
    ----------
    start_stage: int
        The stage index to start from, default 0
    end_stage: int
        The stage index to end at, default None (run all stages)

    :group: core

    """

    def __init__(self, base_dir, name="Pipeline"):
        """
        Constructor

        Parameters
        ----------
        base_dir: str
            The base directory
        name: str
            The name
        """
        super().__init__(name)
        self.start_stage = 0
        self.end_stage = None

        self._base_dir = Path(base_dir)
        self.__idx = -1

    @abstractmethod
    def stages(self):
        """
        Get the stage names

        Returns
        -------
        snms: list of str
            The stage names

        """
        pass

    @property
    def base_dir(self):
        """
        Get the base directory

        Returns
        -------
        Path :
            The base directory

        """
        return self._base_dir

    @property
    def n_stages(self):
        """
        Get the number of stages

        Returns
        -------
        int :
            The number of stages

        """
        return len(self.stages())

    @property
    def stage_index(self):
        """
        Get the current stage index

        Returns
        -------
        int :
            The current stage index

        """
        return self.__idx

    def get_stage_dir(self, stage_index):
        """
        Get the directory for a stage

        Parameters
        ----------
        stage_index: int
            The stage index

        Returns
        -------
        Path :
            The stage directory

        """
        stage = self.stages()[stage_index]
        return self.base_dir / f"{stage_index:02d}_{stage}"

    def __iter__(self):
        """ Get an iterator object for the pipeline. """
        self.__idx = self.start_stage - 1
        return self

    def __next__(self):
        """
        Get the data for the next stage.

        Returns
        -------
        stage_index: int
            The stage index
        stage_name: str
            The stage name
        stage_dir: Path
            The stage directory

        """
        self.__idx += 1
        if self.__idx >= self.n_stages or (
            self.end_stage is not None and 
            self.__idx >= self.end_stage
        ):
            raise StopIteration

        stage_name = self.stages()[self.__idx]
        stage_dir = self.get_stage_dir(self.__idx)

        return self.__idx, stage_name, stage_dir
    
    @abstractmethod
    def run_stage(
        self, 
        stage_index, 
        stage_name, 
        stage_dir, 
        prev_results=None, 
        verbosity=1,
    ):
        """
        Run a stage of the pipeline.

        Parameters
        ----------
        stage_index: int
            The stage index
        stage_name: str
            The stage name
        stage_dir: Path
            The stage directory
        prev_results: object, optional
            The results from the previous stage
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        success: bool
            Whether the stage was successful
        results: object
            The stage results

        """
        pass

    def run(self, start_stage=0, end_stage=None, verbosity=1):
        """
        Run the pipeline.

        Parameters
        ----------
        start_stage: int
            The stage index to start from
        end_stage: int, optional
            The stage index to end at, default None (run all stages)
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        success: bool
            Whether all stages were successful
        results: object
            The final stage results

        """
        hstart = self.start_stage
        hend = self.end_stage
        self.start_stage = start_stage
        self.end_stage = end_stage
        
        success = None
        results = None
        for stage_index, stage_name, stage_dir in self:
            if verbosity > 0:
                print(f"{self.name}: Running stage {stage_index}: {stage_name}")
            try:
                success, results = self.run_stage(
                    stage_index, stage_name, stage_dir, prev_results=results, verbosity=verbosity
                )
                if not success:
                    print(f"{self.name}: Stage {stage_name} failed, stopping pipeline")
                    break
            except Exception as e:
                print(f"{self.name}: Exception occurred during pipeline execution at step {stage_index}: {stage_name}")
                success = False
                raise e

        self.start_stage = hstart
        self.end_stage = hend
        
        return success, results
    