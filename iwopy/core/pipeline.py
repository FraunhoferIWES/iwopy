from abc import ABCMeta, abstractmethod
from pathlib import Path

from .base import Base


class PipelineStage(Base, metaclass=ABCMeta):
    """
    Abstract base class for a pipeline stage.

    A pipeline stage is a single step in an optimization pipeline.

    :group: core

    """

    def initialize(self, pipeline):
        """
        Initialize the stage. This method is called before running the stage.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline this stage belongs to

        """

        i = pipeline.find_stage(self.name)
        assert i >= 0, f"{self.name}: stage not found in pipeline '{pipeline.name}'"

        self.__stage_i = i
        self.__base_dir = pipeline.base_dir
        self.__stage_dir = self.__base_dir / f"{i:02d}_{self.name}"
        self.__stage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index(self):
        """
        Get the stage index in the pipeline

        Returns
        -------
        int :
            The stage index in the pipeline

        """
        return self.__stage_i

    @property
    def base_dir(self):
        """
        Get the base directory

        Returns
        -------
        Path :
            The base directory

        """
        return self.__base_dir

    @property
    def stage_dir(self):
        """
        Get the stage directory

        Returns
        -------
        Path :
            The stage directory

        """
        return self.__stage_dir

    @abstractmethod
    def run(self, prev_stage=None, prev_results=None, verbosity=1):
        """
        Run the pipeline stage.

        Parameters
        ----------
        prev_stage: PipelineStage, optional
            The previous stage
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


class Pipeline(Base):
    """
    Base class for optimization pipelines.

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

    def __init__(self, base_dir, **kwargs):
        """
        Constructor

        Parameters
        ----------
        base_dir: str
            The base directory
        kwargs: dict
            Additional keyword arguments for the base class

        """
        super().__init__(**kwargs)
        self.start_stage = 0
        self.end_stage = None

        self.__stages = []
        self.__base_dir = Path(base_dir)
        self.__idx = -1
        self.__running = False

    def add_stage(self, stage):
        """
        Add a stage to the pipeline

        Parameters
        ----------
        stage: PipelineStage
            The stage to add

        """
        assert not self.running, (
            f"{self.name}: cannot add stage '{stage.name}' while pipeline is running"
        )
        assert isinstance(stage, PipelineStage), (
            f"{self.name}: stage must be an instance of PipelineStage, got {type(stage)}"
        )
        assert stage.name not in self.stage_names, (
            f"{self.name}: stage name '{stage.name}' already exists in pipeline: {self.stage_names}"
        )
        self.__stages.append(stage)

    @property
    def running(self):
        """
        Get whether the pipeline is currently running

        Returns
        -------
        bool :
            Whether the pipeline is currently running

        """
        return self.__running

    @property
    def stage_names(self):
        """
        Get the stage names

        Returns
        -------
        list of str :
            The stage names

        """
        return [stage.name for stage in self.__stages]

    @property
    def base_dir(self):
        """
        Get the base directory

        Returns
        -------
        Path :
            The base directory

        """
        return self.__base_dir

    @property
    def n_stages(self):
        """
        Get the number of stages

        Returns
        -------
        int :
            The number of stages

        """
        return len(self.__stages)

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

    def find_stage(self, stage_name):
        """
        Find the index of a stage by name

        Parameters
        ----------
        stage_name: str
            The stage name

        Returns
        -------
        int :
            The stage index, or -1 if not found

        """
        for i, stage in enumerate(self.__stages):
            if stage.name == stage_name:
                return i
        return -1

    def get_stage(self, stage_index):
        """
        Get a stage by index

        Parameters
        ----------
        stage_index: int
            The stage index

        Returns
        -------
        PipelineStage :
            The stage at the given index

        """
        return self.__stages[stage_index]

    def __iter__(self):
        """Get an iterator object for the pipeline."""
        assert not self.running, (
            f"{self.name}: cannot iterate over pipeline while it is running"
        )
        self.__running = True
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
        assert self.running, (
            f"{self.name}: cannot get next stage data while pipeline is not running"
        )

        self.__idx += 1
        if self.__idx >= self.n_stages or (
            self.end_stage is not None and self.__idx >= self.end_stage
        ):
            self.__running = False
            raise StopIteration

        return self.get_stage(self.__idx)

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
        prev_stage = None
        for stage in self:
            if verbosity > 0:
                print(f"{self.name}: Running stage {stage.index}: {stage.name}")
            try:
                success, results = stage.run(
                    prev_stage=prev_stage,
                    prev_results=results,
                    verbosity=verbosity,
                )
                if not success:
                    print(f"{self.name}: Stage {stage.name} failed, stopping pipeline")
                    break
            except Exception as e:
                print(
                    f"{self.name}: Exception occurred during pipeline execution at step {stage.index}: {stage.name}"
                )
                success = False
                self.__running = False
                raise e

        self.start_stage = hstart
        self.end_stage = hend

        return success, results
