from typing import Any, TypeVar, overload


_T = TypeVar("_T")


def all_subclasses(cls: type[_T]) -> set[type[_T]]:
    """
    Searches all classes derived from some
    base class.

    Parameters
    ----------
    cls
        The base class

    Returns
    -------
    classes
        The derived classes
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


@overload
def new_cls(base_cls: type[_T], cls_name: str) -> type[_T]: ...


@overload
def new_cls(base_cls: type[_T], cls_name: None) -> None: ...


def new_cls(base_cls: type[_T], cls_name: str | None) -> type[_T] | None:
    """
    Run-time class selector.

    Parameters
    ----------
    base_cls
        The base class
    cls_name
        Name of the class

    Returns
    -------
    cls
        The derived class
    """

    if cls_name is None:
        return None

    allc = all_subclasses(base_cls)
    for scls in allc:
        if scls.__name__ == cls_name:
            return scls

    estr = f"Class '{cls_name}' not found, available classes derived from '{base_cls.__name__}' are \n {sorted([i.__name__ for i in allc])}"
    raise KeyError(estr)


@overload
def new_instance(
    base_cls: type[_T], cls_name: str, *args: Any, **kwargs: Any
) -> _T: ...


@overload
def new_instance(
    base_cls: type[_T], cls_name: None, *args: Any, **kwargs: Any
) -> None: ...


def new_instance(
    base_cls: type[_T], cls_name: str | None, *args: Any, **kwargs: Any
) -> _T | None:
    """
    Run-time factory.

    Parameters
    ----------
    base_cls
        The base class
    cls_name
        Name of the class
    args
        Additional parameters for the constructor
    kwargs
        Additional parameters for the constructor

    Returns
    -------
    obj
        The instance of the derived class
    """

    cls = new_cls(base_cls, cls_name)
    if cls is None:
        return None
    else:
        return cls(*args, **kwargs)
