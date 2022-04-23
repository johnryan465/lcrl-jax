class LinearTemporalLogic:
    def __init__(self, formula):
        pass

    @classmethod
    def from_file(cls, filename):
        """
        Reads a LTL formula from a file.
        """
        with open(filename, 'r') as f:
            return cls.from_string(f.read())

    @classmethod
    def from_string(cls, string):
        """
        Reads a LTL formula from a string.
        """
        return cls(string)

    def as_automata(self):
        """
        Returns an automata that accepts the LTL formula.
        """
        pass
