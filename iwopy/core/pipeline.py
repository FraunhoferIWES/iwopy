from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import Base


class PipelineStage(Base, metaclass=ABCMeta):
    """
    Abstract base class for a pipeline stage.

    A pipeline stage is a single step in an optimization pipeline.
    """

    def initialize(self, pipeline: Pipeline | int = 0, verbosity: int = 0) -> None:
        """
        Initialize the stage. This method is called before running the stage.

        Parameters
        ----------
        pipeline
            The pipeline this stage belongs to
        verbosity
            The verbosity level, 0 = silent
        """

        if isinstance(pipeline, int):
            super().initialize(verbosity=verbosity or pipeline)
            return

        i = pipeline.find_stage(self.name)
        assert i >= 0, f"{self.name}: stage not found in pipeline '{pipeline.name}'"

        self.__stage_i: int = i
        self.__base_dir: Path = pipeline.base_dir
        self.__stage_dir: Path = self.__base_dir / f"{i:02d}_{self.name}"
        self.__stage_dir.mkdir(parents=True, exist_ok=True)

        super().initialize(verbosity=verbosity)

    @property
    def index(self) -> int:
        """
        Get the stage index in the pipeline

        Returns
        -------
        index
            The stage index in the pipeline
        """
        return self.__stage_i

    @property
    def base_dir(self) -> Path:
        """
        Get the base directory

        Returns
        -------
        base_dir
            The base directory
        """
        return self.__base_dir

    @property
    def stage_dir(self) -> Path:
        """
        Get the stage directory

        Returns
        -------
        stage_dir
            The stage directory
        """
        return self.__stage_dir

    @abstractmethod
    def run(
        self,
        prev_stage: PipelineStage | None = None,
        prev_results: object | None = None,
        verbosity: int = 1,
        **kwargs: object,
    ) -> tuple[bool, object | None]:
        """
        Run the pipeline stage.

        Parameters
        ----------
        prev_stage
            The previous stage
        prev_results
            The results from the previous stage
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Additional parameters for the pipeline stage

        Returns
        -------
        success
            Whether the stage was successful
        results
            The stage results
        """

    def finalize(self, pipeline: Pipeline | int = 0, verbosity: int = 0) -> None:
        """
        Finalize the stage. This method is called after running the stage.

        Parameters
        ----------
        pipeline
            The pipeline this stage belongs to
        verbosity
            The verbosity level, 0 = silent
        """
        if isinstance(pipeline, int):
            return super().finalize(verbosity or pipeline)
        return super().finalize(verbosity)


class Pipeline(Base):
    """Base class for optimization pipelines.

    An optimization pipeline is a collection of optimization problems
    and optimizers that are run one after another. Each step
    of this process is called a stage.
    """

    def __init__(self, base_dir: str | Path, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        base_dir
            The base directory
        kwargs
            Additional keyword arguments for the base class
        """
        super().__init__(**kwargs)
        self.start_stage: int = 0
        self.end_stage: int | None = None

        self.__stages: list[PipelineStage] = []
        self.__base_dir: Path = Path(base_dir)
        self.__idx: int = -1
        self.__running: bool = False

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline

        Parameters
        ----------
        stage
            The stage to add
        """
        assert not self.initialized, (
            f"{self.name}: cannot add stage '{stage.name}' after pipeline has been initialized"
        )
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

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """
        assert not self.running, (
            f"{self.name}: cannot initialize pipeline while it is running"
        )
        assert len(self.__stages) > 0, (
            f"{self.name}: pipeline must have at least one stage, use function 'add_stage' to add stages to the pipeline"
        )

        for stage in self.__stages:
            stage.initialize(self, verbosity=verbosity)

        super().initialize(verbosity=verbosity)

    @property
    def running(self) -> bool:
        """
        Get whether the pipeline is currently running

        Returns
        -------
        running
            Whether the pipeline is currently running
        """
        return self.__running

    @property
    def stage_names(self) -> list[str]:
        """
        Get the stage names

        Returns
        -------
        stage_names
            The stage names
        """
        return [stage.name for stage in self.__stages]

    @property
    def base_dir(self) -> Path:
        """
        Get the base directory

        Returns
        -------
        base_dir
            The base directory
        """
        return self.__base_dir

    @property
    def n_stages(self) -> int:
        """
        Get the number of stages

        Returns
        -------
        n_stages
            The number of stages
        """
        return len(self.__stages)

    @property
    def stage_index(self) -> int:
        """
        Get the current stage index

        Returns
        -------
        stage_index
            The current stage index
        """
        return self.__idx

    def find_stage(self, stage_name: str) -> int:
        """
        Find the index of a stage by name

        Parameters
        ----------
        stage_name
            The stage name

        Returns
        -------
        stage_index
            The stage index, or -1 if not found
        """
        for i, stage in enumerate(self.__stages):
            if stage.name == stage_name:
                return i
        return -1

    def get_stage(self, stage_index: int) -> PipelineStage:
        """
        Get a stage by index

        Parameters
        ----------
        stage_index
            The stage index

        Returns
        -------
        stage
            The stage at the given index
        """
        return self.__stages[stage_index]

    def __iter__(self) -> Iterator[PipelineStage]:
        """Get an iterator object for the pipeline."""
        assert self.initialized, (
            f"{self.name}: cannot iterate over pipeline before it has been initialized"
        )
        assert not self.running, (
            f"{self.name}: cannot iterate over pipeline while it is running"
        )
        self.__running = True
        self.__idx = self.start_stage - 1
        return self

    def __next__(self) -> PipelineStage:
        """
        Get the data for the next stage.

        Returns
        -------
        stage_index
            The stage index
        stage_name
            The stage name
        stage_dir
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

    def run(
        self,
        start_stage: int = 0,
        end_stage: int | None = None,
        initial_results: object | None = None,
        finalize: bool = True,
        verbosity: int = 1,
        **kwargs: object,
    ) -> tuple[bool | None, object | None]:
        """Run a selected half-open range of pipeline stages.

        The first selected stage receives ``initial_results`` and the stage
        registered immediately before it as ``prev_stage``. Execution stops
        after a stage returns ``success=False``. The pipeline always leaves its
        running state before returning or propagating an exception.

        Parameters
        ----------
        start_stage
            Index of the first stage to run.
        end_stage
            Exclusive end index, or ``None`` to run through the final stage.
        initial_results
            Application-defined results supplied to the first selected stage,
            for example a persisted result used with ``start_stage > 0``.
        finalize
            Whether to finalize initialized stages after normal completion or
            a stage-reported failure.
        verbosity
            Verbosity level, where zero is silent except for propagated stage
            errors.
        kwargs
            Additional keyword arguments passed to every selected stage.

        Returns
        -------
        success
            Whether every selected stage succeeded, or ``None`` when no stage
            ran.
        results
            Results returned by the final stage that ran, or
            ``initial_results`` when no stage ran.
        """
        assert not self.running, f"{self.name}: cannot run pipeline while it is running"

        if not self.initialized:
            self.initialize(verbosity=verbosity)

        hstart = self.start_stage
        hend = self.end_stage
        self.start_stage = start_stage
        self.end_stage = end_stage

        success = None
        results = initial_results
        prev_stage = self.get_stage(start_stage - 1) if start_stage > 0 else None
        try:
            for stage in self:
                if verbosity > 0:
                    print(f"{self.name}: Running stage {stage.index}: {stage.name}")
                success, results = stage.run(
                    prev_stage=prev_stage,
                    prev_results=results,
                    verbosity=verbosity,
                    **kwargs,
                )
                if not success:
                    print(f"{self.name}: Stage {stage.name} failed, stopping pipeline")
                    break
                prev_stage = stage
        except Exception:
            print(
                f"{self.name}: Exception occurred during pipeline execution at step {stage.index}: {stage.name}"
            )
            success = False
            raise
        finally:
            self.__running = False
            self.start_stage = hstart
            self.end_stage = hend

        if finalize:
            self.finalize(verbosity=verbosity)

        return success, results

    def finalize(self, verbosity: int = 0) -> None:
        """
        Finalize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """
        assert not self.running, (
            f"{self.name}: cannot finalize pipeline while it is running"
        )

        for stage in self.__stages:
            stage.finalize(self, verbosity=verbosity)

        return super().finalize(verbosity=verbosity)
