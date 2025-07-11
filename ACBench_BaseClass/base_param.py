from dataclasses import dataclass, asdict

class BaseParam:

    #... ...
    #add param
    #... ...

    @classmethod
    def from_args(cls, args):
        instance = cls()
        for key in cls.__annotations__.keys():
            if hasattr(args, key):
                setattr(instance, key, getattr(args, key))
        return instance