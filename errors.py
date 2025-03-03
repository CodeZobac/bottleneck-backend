class BottleneckError(Exception):
    """Base exception for Bottleneck application."""
    pass

class ComponentDataError(BottleneckError):
    """Error related to component data processing."""
    pass

class FileNotFoundError(BottleneckError):
    """Error when a required file is not found."""
    pass

class DataFormatError(BottleneckError):
    """Error when data format is incorrect."""
    pass
