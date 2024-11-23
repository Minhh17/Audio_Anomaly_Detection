class AverageMeter():
    def __init__(self) -> None:
        self.accumulate_sum = 0
        self.rounds = 0

    def accumulate_stat(self, value):
        self.accumulate_sum += value
        self.rounds += 1

    def average_stat(self):
        return self.accumulate_sum/self.rounds

    def reset(self):
        self.accumulate_sum = 0
        self.rounds = 0