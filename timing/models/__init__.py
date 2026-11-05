"""Model layer. Any module in this package is imported so its models register themselves."""

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    if _module.name not in {"base", "registry"}:
        importlib.import_module(f"{__name__}.{_module.name}")
