class PPrintable():
    def __str__(self):
        from pprint import pformat
        return pformat(self.__dict__, indent=4)

    def __repr__(self):
        from pprint import pformat
        return pformat(self.__dict__, indent=8)