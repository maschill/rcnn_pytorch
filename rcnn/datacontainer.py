def abstract_variables(*args):
    """enforces that instances variables are set
    https://stackoverflow.com/questions/56997102/python-notimplementederror-for-instance-attributes
    """

    class av:
        def __init__(self, error_message):
            self.error_message = error_message

        def __get__(self, *args, **kwargs):
            raise NotImplementedError(self.error_message)

    def f(klass):
        for arg in args:
            setattr(klass, arg, av("Descendants must set variable `{}`".format(arg)))
        return klass

    return f


@abstract_variables("train_loader", "val_loader", "test_loader", "sizes", "loader_dict")
class DataContainer:
    """base class for dataloaders"""

    def dl_dict(self,):
        raise NotImplementedError

    def update_val_loader(self, n, mode="cutout"):
        raise NotImplementedError
